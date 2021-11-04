import time
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
import os
import os.path as osp

from evaluator import Evaluator

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

reg_criterion = torch.nn.NLLLoss()
#reg_criterion = torch.nn.BCEWithLogitsLoss()

def get_dataloaders(dataset, split_idx, batch_size):

    train_dataset = dataset[split_idx["train"]]
    test_dataset = dataset[split_idx["test"]]
    val_dataset = dataset[split_idx["valid"]]

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def test(dataset, split_idx, model, evaluator,  epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size, weight_decay, logger=None, log_dir = None, checkpoint_dir = None, checkpoint_name="checkpoint.pt"):

    print("dataset", dataset)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    train_loader, val_loader, test_loader = get_dataloaders(dataset,split_idx, batch_size)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if checkpoint_dir is not '':
        os.makedirs(checkpoint_dir, exist_ok=True)

    if log_dir is not '':
        os.makedirs(log_dir, exist_ok=True)
        writer_train = SummaryWriter(log_dir= os.path.join(log_dir, "train"))
        writer_val = SummaryWriter(log_dir= os.path.join(log_dir, "val"))
        writer_test = SummaryWriter(log_dir= os.path.join(log_dir, "test"))

    t_start = time.perf_counter()
    val_loss, acc, durations = 0,0,0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader)
        _, train_dict = eval(model, train_loader, evaluator)
        val_loss, val_dict = eval(model, val_loader, evaluator)
        _, test_dict = eval(model, test_loader, evaluator)
        eval_info = {

            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val' : val_dict,
            'test': test_dict,
            'train' : train_dict
        }

        if log_dir is not '':

            writer_val.add_scalar('loss', val_loss, epoch)
            writer_train.add_scalar('loss', train_loss, epoch)

            writer_train.add_scalar('acc', train_dict["acc"], epoch)
            writer_val.add_scalar('acc', val_dict["acc"], epoch)
            writer_test.add_scalar('acc', test_dict["acc"], epoch)

            writer_train.add_scalar('mcc', train_dict["mcc"], epoch)
            writer_val.add_scalar('mcc', val_dict["mcc"], epoch)
            writer_test.add_scalar('mcc', test_dict["mcc"], epoch)

            writer_train.add_scalar('f1', train_dict["f1"], epoch)
            writer_val.add_scalar('f1', val_dict["f1"], epoch)
            writer_test.add_scalar('f1', test_dict["f1"], epoch)

        if logger is not None:
            print(eval_info)#print(epoch, 'Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.3f},  Val Acc: {:.3f}, Test Acc: {:.3f},Train Mcc: {:.3f},  Val Mcc: {:.3f}, Test Mcc: {:.3f}'.format(train_loss, val_loss, train_dict["acc"], val_dict["acc"],test_dict['acc'],train_dict["mcc"], val_dict["mcc"],test_dict['mcc']))

        best_val_acc = 0
        acc = val_dict["mcc"]
        if acc > best_val_acc:
            best_val_acc = acc
            best_info = eval_info
        #     if checkpoint_dir is not None:
        #         print('Saving checkpoint...')
        #         checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
        #                       'optimizer_state_dict': optimizer.state_dict(),
        #                       'scheduler_state_dict': scheduler.state_dict(), 'best_test_acc': best_val_acc,
        #                       'num_params': num_params}
        #         torch.save(checkpoint, osp.join(checkpoint_dir,checkpoint_name))

        scheduler.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations = t_end - t_start
    if log_dir is not '':
        writer_val.close()
        writer_test.close()
        writer_train.close()

    print(best_info)
    print('Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.3f},  Val Acc: {:.3f}, Test Acc: {:.3f},Train Mcc: {:.3f},  Val Mcc: {:.3f}, Test Mcc: {:.3f}, Duration: {:.3f}'.
          format( best_info["train_loss"],  best_info["val_loss"],best_info["train"]["acc"], best_info["val"]["acc"],  best_info["test"]["acc"] ,best_info["train"]["mcc"], best_info["val"]["mcc"],  best_info["test"]["mcc"], durations))

    return best_info



def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for  data in loader:

        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = reg_criterion(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)



def eval(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []
    loss_all = 0
    for  data in loader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
        loss = reg_criterion(pred, data.y.view(-1))
        loss_all += data.y.size(0) * loss.item()

        y_true.append(data.y.view(-1).detach().cpu())
        y_pred.append(pred.max(1)[1].detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred":y_pred}

    return loss_all / len(loader.dataset), evaluator.eval(input_dict)

################

#
# def cross_validation_with_val_set(dataset, dsname, model, folds, epochs, batch_size,
#                                   lr, lr_decay_factor, lr_decay_step_size,
#                                   weight_decay, logger=None):
#
#     evals , val_losses, f1s, mccs, accs, durations = [], [], [], [], [], []
#     for fold, (train_idx, test_idx,
#                val_idx) in enumerate(zip(*k_fold(dataset, dsname, folds))):
#
#
#
#         train_dataset = dataset[train_idx]
#         test_dataset = dataset[test_idx]
#         val_dataset = dataset[val_idx]
#         evaluator = Evaluator()
#
#         train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
#         test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
#
#         model.to(device).reset_parameters()
#         optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#         scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
#
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#
#         t_start = time.perf_counter()
#
#         if dsname != "JT":
#             reg_criterion = torch.nn.BCEWithLogitsLoss()
#
#         for epoch in range(1, epochs + 1):
#             train_loss = train(model, optimizer, train_loader)
#             _, train_dict = eval(model, train_loader, evaluator)
#             val_loss, val_dict = eval(model, val_loader, evaluator)
#             _, test_dict = eval(model, test_loader, evaluator)
#
#             val_losses.append(val_loss)
#             accs.append(test_dict["acc"])
#             mccs.append(test_dict["mcc"])
#
#             eval_info = {
#
#                 'epoch': epoch,
#                 'fold' : fold,
#
#                 'train_loss': train_loss,
#                 'val_loss': val_loss,
#                 'val': val_dict,
#                 'test': test_dict,
#                 'train': train_dict
#             }
#
#             if logger is not None:
#                 print(eval_info)
#
#             scheduler.step()
#
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#
#         t_end = time.perf_counter()
#         durations.append(t_end - t_start)
#
#     loss, acc, mcc,  duration = tensor(val_losses), tensor(accs), tensor(mccs),tensor(durations)
#     loss, acc, mcc  = loss.view(folds, epochs), acc.view(folds, epochs),mcc.view(folds, epochs)
#     loss, argmin = loss.min(dim=1)
#     acc = acc[torch.arange(folds, dtype=torch.long), argmin]
#     mcc = mcc[torch.arange(folds, dtype=torch.long), argmin]
#
#     loss_mean = loss.mean().item()
#     acc_mean = acc.mean().item()
#     acc_std = acc.std().item()
#
#     mcc_mean = mcc.mean().item()
#     mcc_std = mcc.std().item()
#
#     duration_mean = duration.mean().item()
#     print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Test MCC: {:.3f} ± {:.3f}, Duration: {:.3f}'.
#           format(loss_mean, acc_mean, acc_std, mcc_mean, mcc_std, duration_mean))
#     result = {
#
#         'val_loss': loss_mean,
#         'acc_mean': acc_mean,
#         'acc_std': acc_std,
#         "mcc_mean" : mcc_mean,
#         "mcc_std" : mcc_std
#     }
#     return result
#
#
# def k_fold(dataset, dsname, folds):
#     skf = StratifiedKFold(folds, shuffle=True, random_state=12345)
#
#     test_indices, train_indices = [], []
#     if dsname == "JT":
#         kfolds = skf.split(torch.zeros(len(dataset)), dataset.data.y)
#     else:
#         print(dataset.data.y)
#         kfolds = skf.split(torch.zeros(len(dataset)), dataset.data.y.argmax(1))
#
#     for _, idx in kfolds:
#         test_indices.append(torch.from_numpy(idx).to(torch.long))
#
#     val_indices = [test_indices[i - 1] for i in range(folds)]
#
#     for i in range(folds):
#         train_mask = torch.ones(len(dataset), dtype=torch.bool)
#         train_mask[test_indices[i]] = 0
#         train_mask[val_indices[i]] = 0
#         train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
#
#     return train_indices, test_indices, val_indices
