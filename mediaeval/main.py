import json
from itertools import product

import argparse

from evaluator import Evaluator

from utils import get_dataset_local,get_split_idx_kfold
import train_classifier
import train_multilabel_classifier
from  models.gin import  GIN



parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr_decay_factor', type=float, default=0.2)
parser.add_argument('--lr_decay_step_size', type=int, default=10)

parser.add_argument('--log_dir', type=str, default="/usr/src/temp/logs/")
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--dataset', type=str, default="mediaeval")
parser.add_argument('--checkpoint_dir', type=str, default="/usr/src/temp/checkpoint/")
args = parser.parse_args()

#TODO: add all to args and config file & copy config file to results folder to track the config
windows = [3]
layers = [1,2,3]
hiddens = [64,128, 256]
nets = [GIN]

results = []
dataset_name = args.dataset
task = "task-2"


isMultiLabel = False
if task == 'task-2':
    isMultiLabel = True

def logger(info):
    epoch =  info['epoch']
    train_loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['test_acc']
    print('{:03d}:Train Loss: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
         epoch, train_loss, val_loss, test_acc))



if isMultiLabel:
    run_funct = train_multilabel_classifier.run
results = []
for window in windows:
    print("Generating dataset for window size ", window)
    dataset = get_dataset_local(dataset_name, task, windows=window)
    split_idx = get_split_idx_kfold(dataset, args.kfold, isMultiLabel=isMultiLabel)

    for Net in nets:
        best_acc = 0 # (loss, acc, std)
        best_result = None
        print('-----\n{} - {}'.format(dataset_name, Net.__name__))

        for num_layers, hidden in product(layers, hiddens):

            model = Net(dataset.num_classes, dataset.num_features, num_layers, hidden)

            print('Num layers {} - hiddens {}'.format(num_layers, hidden))

            ### automatic evaluator.
            name =  "_".join([dataset_name, str(num_layers),str(hidden)])
            print(name)
            evaluator = Evaluator()

            for fold, split in enumerate(split_idx):

                best_info = run_funct(
                    dataset,
                    split,
                    model,
                    evaluator,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    lr_decay_factor=args.lr_decay_factor,
                    lr_decay_step_size=args.lr_decay_step_size,
                    weight_decay=0,
                    logger=logger,
                    log_dir=args.log_dir + task + "/" + name + "_" +  str(window),
                    checkpoint_dir=args.checkpoint_dir + "/"+ task + "/",
                    checkpoint_name= name + "_" + str(window) + ".pt"
                )


                best_info["num_layers"] = num_layers
                best_info["hidden_layers"] = hidden
                best_info["windows"] = window
                best_info["fold"] = fold
                print('{} - {}: {}'.format(dataset_name, model, best_info))
                results.append(best_info)


with open(args.log_dir + "read.me","w+") as r:
    json.dump(args.__dict__, r, indent=2)


print("##################")
for r in results:
    print(r)

