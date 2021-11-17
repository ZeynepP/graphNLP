import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os

from utils import num_graphs, get_dataloaders, calculate_multilabel_weights

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
pos_weigths_mediaeval = torch.Tensor([0.021879021879022,	0.090733590733591,	0.122265122265122	,0.146074646074646	,0.148005148005148,0.036679536679537	,0.087516087516088	,0.079150579150579,	0.05019305019305])
pos_weigths_mediaeval = pos_weigths_mediaeval.to(device)
reg_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')



def run(dataset, split_idx, model, evaluator,  epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size, weight_decay, logger=None, log_dir = None, checkpoint_dir = None, checkpoint_name="checkpoint.pt"):

    print("dataset", dataset)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    train_loader, val_loader = get_dataloaders(dataset,split_idx, batch_size)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    if torch.cuda.is_available():
        torch.cuda.synchronize(device=device)

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
       # _, test_dict = eval(model, test_loader, evaluator)
        eval_info = {

            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val' : val_dict,
     #       'test': test_dict,
            'train' : train_dict
        }

        if log_dir is not '':

            writer_val.add_scalar('loss', val_loss, epoch)
            writer_train.add_scalar('loss', train_loss, epoch)

            writer_train.add_scalar('acc', train_dict["acc"], epoch)
            writer_val.add_scalar('acc', val_dict["acc"], epoch)
            #    writer_test.add_scalar('acc', test_dict["acc"], epoch)

            writer_train.add_scalar('mcc', train_dict["mcc"], epoch)
            writer_val.add_scalar('mcc', val_dict["mcc"], epoch)
            #    writer_test.add_scalar('mcc', test_dict["mcc"], epoch)

            writer_train.add_scalar('f1', train_dict["f1"], epoch)
            writer_val.add_scalar('f1', val_dict["f1"], epoch)
            #    writer_test.add_scalar('f1', test_dict["f1"], epoch)

        if logger is not None:
            print(eval_info)#print(epoch, 'Train Loss: {:.4f}, Val Loss: {:.4f}, Train Acc: {:.3f},  Val Acc: {:.3f}, Test Acc: {:.3f},Train Mcc: {:.3f},  Val Mcc: {:.3f}, Test Mcc: {:.3f}'.format(train_loss, val_loss, train_dict["acc"], val_dict["acc"],test_dict['acc'],train_dict["mcc"], val_dict["mcc"],test_dict['mcc']))

        best_val_acc = 0
        acc = val_dict["acc"]
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
        torch.cuda.synchronize(device=device)

    t_end = time.perf_counter()

    if log_dir is not '':
        writer_val.close()
        # writer_test.close()
        writer_train.close()

    return best_info




def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for  data in loader:

        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)

        loss = reg_criterion(out, data.y) #.view(-1)
        loss = (loss * pos_weigths_mediaeval).mean()
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
            logits =  model(data)

        loss = reg_criterion(logits, data.y) #
        loss = (loss * pos_weigths_mediaeval).mean()
        loss_all += loss.item() * num_graphs(data)

        # since we're using BCEWithLogitsLoss, to get the predictions -
        # - sigmoid has to be applied on the logits first
     #
     #
        logits = torch.sigmoid(logits)
     #    # print(logits)
        logits = logits>0.10
        #  logits = np.round(logits.cpu().numpy() + 0.41)
     #    # print(logits)
     #
     # #   print(labels[0], logits[0])
     #    # the tensors are detached from the gpu and put back on -
     #    # - the cpu, and then converted to numpy in order to -
     #    # - use sklearn's metrics.
        y_pred.append(logits.detach().cpu())
        y_true.append(data.y.detach().cpu())


    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred":y_pred}

    return loss_all / len(loader.dataset),  evaluator.eval_sklearn(input_dict)


