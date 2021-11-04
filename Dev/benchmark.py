from itertools import product

import argparse

from evaluator import Evaluator
from utils import get_dataset_local, get_split_idx,get_split_idx_kfold
from train_eval import test

from models.gcn import GCN, GCNWithJK

from  models.gin import GIN0, GIN0WithJK, GIN, GINWithJK

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=10)
parser.add_argument('--log_dir', type=str, default="/usr/src/temp/logs/")
parser.add_argument('--kfold', type=int, default=1)
parser.add_argument('--dataset', type=str, default="mediaeval")
parser.add_argument('--checkpoint_dir', type=str, default="/usr/src/temp/checkpoint/")
args = parser.parse_args()


layers = [1,2,3]
hiddens = [64,128,256]
nets = [GIN]


results = []
dataset_name = "mediaeval"
task = "task-1"
dataset = get_dataset_local(dataset_name, task)


def logger(info):
    epoch =  info['epoch']
    train_loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['test_acc']
    print('{:03d}:Train Loss: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
         epoch, train_loss, val_loss, test_acc))


split_idx = get_split_idx_kfold(dataset,args.kfold)

for Net in nets:
    best_acc = 0 # (loss, acc, std)
    best_result = None
    print('-----\n{} - {}'.format(dataset_name, Net.__name__))

    for num_layers, hidden in product(layers, hiddens):

        model = Net(dataset, num_layers, hidden)

        print('Num layers {} - hiddens {}'.format(num_layers, hidden))

        ### automatic evaluator.
        name =  "_".join([dataset_name, str(num_layers), "ggnlayer_", str(hidden),"_hiddenlayers"])
        evaluator = Evaluator()

        for fold, split in enumerate(split_idx):

            best_info = test(
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
                log_dir=args.log_dir + name,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_name= name + ".pt"
            )


            best_info["num_layers"] = num_layers
            best_info["hidden_layers"] = hidden
            best_info["fold"] = fold
            print('{} - {}: {}'.format(dataset_name, model, best_info))
            results += ['{} - {}: {}'.format(dataset_name, model, best_result)]
print('BEST MODEL -----\n{}'.format('\n'.join(results)))