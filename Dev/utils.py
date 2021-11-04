import os.path as osp
from collections import Counter

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from JTDataset import JTDataset
from JTDatasetJson import JTDatasetJson
from mediaevalDataset import MediaevalDataset
import numpy as np

import nltk
from nltk.corpus import stopwords

import os
os.environ['http_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTP_PROXY'] = "http://firewall.ina.fr:81"
os.environ['https_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTPS_PROXY'] = "http://firewall.ina.fr:81"

def get_split_idx(dataset):

    split_idx = {}
    # this part for test
    ids = [i for i in range(len(dataset))]
    y = dataset.data.y.cpu().detach().numpy()
    rs = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=0)  # replace with StratifiedShuffleSplit
    for train_index, test_index in rs.split(ids, y):
        split_idx["train"], split_idx["test"] = train_index, test_index

    rs = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=0)
    for valid_index, test_index in rs.split(split_idx["test"], y[split_idx["test"]]):
        split_idx["valid"], split_idx["test"] = valid_index, test_index

    c = Counter(y[split_idx["train"]])
    print("train", c)
    c = Counter(y[split_idx["valid"]])
    print("valid", c)
    c = Counter(y[split_idx["test"]])
    print("test", c)

    split_idx["train"] = torch.from_numpy(np.array(split_idx["train"], dtype=np.int64))
    split_idx["test"] = torch.from_numpy(np.array(split_idx["test"], dtype=np.int64))
    split_idx["valid"] = torch.from_numpy(np.array(split_idx["valid"], dtype=np.int64))


    return split_idx

def get_split_idx_kfold(dataset, folds):


    splits = []
    # this part for test
    ids = [i for i in range(len(dataset))]
    y = dataset.data.y.cpu().detach().numpy()
    rs = StratifiedShuffleSplit(n_splits=folds, test_size=.2, random_state=12345)  # replace with StratifiedShuffleSplit
    for train_index, test_index in rs.split(ids, y):
        split_idx = {}
        split_idx["train"], split_idx["test"] = train_index, test_index
        rs = StratifiedShuffleSplit(n_splits=1, test_size=.5, random_state=1245)
        for valid_index, test_index in rs.split(split_idx["test"], y[split_idx["test"]]):
            split_idx["valid"], split_idx["test"] = valid_index, test_index


        c = Counter(y[split_idx["train"]])
        print( "train", c)
        c = Counter(y[split_idx["valid"]])
        print("valid", c)
        c = Counter(y[split_idx["test"]])
        print("test", c)

        split_idx["train"] = torch.from_numpy(np.array(split_idx["train"], dtype=np.int64))
        split_idx["test"] = torch.from_numpy(np.array(split_idx["test"], dtype=np.int64))
        split_idx["valid"] = torch.from_numpy(np.array(split_idx["valid"], dtype=np.int64))
        print(split_idx)
        splits.append(split_idx)

    return splits

def dict2file(dictionary, embeddings, meta_out_file):
    fw = open(meta_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        fw.write(dictionary[i] + "\n")
    fw.close()


def embedding2file(embeddings, embeddings_out_file):
    print("Embedding:", embeddings.shape)
    fw = open(embeddings_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        line = ''
        for j in range(embeddings.shape[1]):
            line = line + str(embeddings[i, j]) + '\t'
        fw.write(line.strip() + '\n')
    fw.close()



class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def get_dataset(name, sparse=True, cleaned=False):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical



def get_dataset_local(name, task=None, sparse=True, cleaned=False):
    if name == "JT":
        dataset = JTDataset(root='/usr/src/temp/data/ina/')
    elif name =="mediaeval":
        dataset = MediaevalDataset(root='/usr/src/temp/data/mediaeval/' + task, task=task)
    elif name == "JT-json":
        dataset = JTDatasetJson(root='/usr/src/temp/data/ina-json/')

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # dataset.data.edge_attr = None
    print(dataset.data.x.dtype,dataset.data.edge_index.dtype)
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset

