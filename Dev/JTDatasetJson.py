import json
import multiprocessing
import os
import os.path as osp
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from transformers import FlaubertModel, FlaubertTokenizer
torch.set_printoptions(edgeitems=1)

import os
os.environ['http_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTP_PROXY'] = "http://firewall.ina.fr:81"
os.environ['https_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTPS_PROXY'] = "http://firewall.ina.fr:81"



class JTDatasetJson(InMemoryDataset):
    def __init__(self, root='data/Ina', transform=None, pre_transform=None):

        self.modelname = 'flaubert/flaubert_base_uncased'
        self.model, self.log = FlaubertModel.from_pretrained(self.modelname, output_loading_info=True)
        self.tokenizer = FlaubertTokenizer.from_pretrained(self.modelname)
        self.original_root = root
        self.folder = root

        self.truncate = True  # whether to truncate long document
        self.MAX_TRUNC_LEN = 400
        self.word_embeddings_dim = 768
        self.cleaned = False
        self.default_emb = np.random.uniform(-0.01, 0.01,
                                             self.word_embeddings_dim)  # empty embedding for non-seen vocab
        super(JTDatasetJson, self).__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])




    @property
    def raw_file_names(self):
        return 'subjects_v2_2017-2019.json'

    @property
    def processed_file_names(self):
        return 'processed.pt'


    def get_graph_data(self, args):
        text, label, id = args

        data = None
        graph = self.text2graph(text)
        # filter graphs without edges
        try:

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])
            data = Data()
            data.__num_nodes__ = int(graph['num_nodes'])
            data.text = text
            data.edge_index = torch.from_numpy(graph['edge_index'])
            data.edge_attr = torch.from_numpy(graph['edge_feat'])
            data.x = torch.from_numpy(graph['node_feat'])
            data.y = torch.Tensor([int(label)]).long()
            data.doc_id = int(id)


        except Exception as ex:
            print(ex)
            return  None
        return  data

    def process(self):
        data_list=[]
        labels_dict = {}
        print(osp.join(self.raw_dir, self.raw_file_names))
        file_path = osp.join(self.raw_dir, self.raw_file_names)
        num = sum(1 for line in open(file_path))
        with open(file_path, encoding='utf-8') as file:
            for line in tqdm(file, total=num):
                json_obj = json.loads(line)
                text_raw = json_obj["document"]
                label_str = json_obj["topics"][0]["label"]
                if label_str in labels_dict:
                    label = labels_dict[label_str]
                else:
                    label = len(labels_dict) + 1
                    labels_dict[label_str] = label
                    print(labels_dict)
                data = self.get_graph_data((text_raw, label, json_obj["id"] ))
                if data is not None:
                    data_list.append(data)
                else:
                    print("Data none for ", line)


        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


    def text2graph(self, text, window_size=3):

            doc_words = text.split()
            doc_len = len(doc_words)

            token_ids = torch.tensor([self.tokenizer.encode(text, truncation=self.truncate)])
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids[0])
            last_layer = self.model(token_ids)[0]

            doc_vocab = list(set(doc_words))
            doc_nodes = len(doc_vocab)

            doc_word_id_map = {}
            for j in range(doc_nodes):
                doc_word_id_map[doc_vocab[j]] = j

            # sliding windows
            windows = []
            if doc_len <= window_size:
                windows.append(doc_words)
            else:
                for j in range(doc_len - window_size + 1):
                    window = doc_words[j: j + window_size]
                    windows.append(window)

            word_pair_count = {}
            for window in windows:
                for p in range(1, len(window)):
                    for q in range(0, p):
                        word_p = window[p]
                        word_p_id = doc_word_id_map[word_p]
                        word_q = window[q]
                        word_q_id = doc_word_id_map[word_q]
                        if word_p_id == word_q_id:
                            continue
                        word_pair_key = (word_p_id, word_q_id)
                        # word co-occurrences as weights
                        if word_pair_key in word_pair_count:
                            word_pair_count[word_pair_key] += 1.
                        else:
                            word_pair_count[word_pair_key] = 1.
                        # bi-direction
                        word_pair_key = (word_q_id, word_p_id)
                        if word_pair_key in word_pair_count:
                            word_pair_count[word_pair_key] += 1.
                        else:
                            word_pair_count[word_pair_key] = 1.

            edge_attr = None
            edge_index = None
            edges_list = []
            edge_features_list = []
            for obj in word_pair_count:
                i = int(obj[0])
                j = int(obj[1])

                weight = float(word_pair_count[obj])

                edges_list.append((i, j))
                edge_feature = [weight]
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
                edge_index = np.array(edges_list).T
                # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
                edge_attr = np.array(edge_features_list, dtype=np.float32)

            embs = []


            for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
                index = None
                if k + "</w>" in tokens:
                    index = tokens.index(k + "</w>")
                elif k in tokens:
                    index = tokens.index(k)

                if index != None:
                    embs.append(last_layer[:, index, :][0].detach().numpy())
                else:
                    embs.append(self.default_emb)

            #

            graph = dict()
            graph['edge_index'] = edge_index
            graph['edge_feat'] = edge_attr
            graph['node_feat'] = np.array(embs, dtype=np.float32)
            graph['num_nodes'] = len(embs)
            return graph

    def text2graph_cleaned(self, text, window_size=3):

        doc_words = text.split()

        if self.truncate:
            doc_words = doc_words[:self.MAX_TRUNC_LEN]
        doc_len = len(doc_words)

        token_ids = torch.tensor([self.tokenizer.encode(" ".join(doc_words))])
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids[0])
        last_layer = self.model(token_ids)[0]

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)


        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j


        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = doc_word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = doc_word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.


        edge_attr = None
        edge_index = None
        edges_list = []
        edge_features_list = []
        for obj in word_pair_count:

            i = int(obj[0])
            j = int(obj[1])

            weight = float(word_pair_count[obj])

            edges_list.append((i, j))
            edge_feature = [weight]
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            edge_index = np.array(edges_list).T
            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.float32)

        embs = []

        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
                index = None
                if k + "</w>" in tokens:
                    index = tokens.index(k + "</w>")
                elif k in tokens:
                    index = tokens.index(k)

                if index != None:
                    embs.append(last_layer[:, index, :][0].detach().numpy())
                else:
                    embs.append(self.default_emb)
#

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = np.array(embs, dtype=np.float32)
        graph['num_nodes'] = len(embs)
        return graph

if __name__ == '__main__':
    dataset = JTDataset(root="./data/ina") #/usr/src/temp/data/ina/
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[5])
    print(dataset[5].y)
