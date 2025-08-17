'''
# 1. recommend to use 4 GPUs for training:

# train DMA_STGCRNN (multi-attribute GCN + GRU) model with 4 GPUs
torchrun --nproc_per_node 4 train_ddp.py --model_name DMA_STGCRNN --log_dir log --pred_len 16 --learning_rate 0.002 --hidden_num 240 --batch_size 512 --epoch 1000 --lrDecay 0.98 --ddp_training 'True'

# train GraphGRU (GCN + GRU) model with 4 GPUs
torchrun --nproc_per_node 4 train_ddp.py --model_name GraphGRU --log_dir log --pred_len 16 --learning_rate 0.002 --hidden_num 240 --batch_size 512 --epoch 1000 --lrDecay 0.98 --ddp_training 'True'

# 2. If you want to use 1 GPU for training, please use the following command:

# train DMA_STGCRNN (multi-attribute GCN + GRU) model with 1 GPU
python train_ddp.py --model_name DMA_STGCRNN --log_dir log --pred_len 16 --learning_rate 0.002 --hidden_num 240 --batch_size 512 --epoch 1000 --lrDecay 0.98 --ddp_training 'False'

# train GraphGRU (GCN + GRU) model with 1 GPU
python train_ddp.py --model_name GraphGRU --log_dir log --pred_len 16 --learning_rate 0.002 --hidden_num 240 --batch_size 512 --epoch 1000 --lrDecay 0.98 --ddp_training 'False'
'''
import models
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from config.GraphGRU_config import default_hyperparameters
import scipy.io
# from torchsummary import summary
from data.PVdataloader import PV_dataloader_func, PV_single_dataloader_func, load_data
import torch.distributed as dist


# main network
class Network(nn.Module):
    def __init__(self, args, adj):
        super(Network, self).__init__()
        self.net = self.get_model(args, adj)

    def forward(self, x):
        y = self.net(x)
        return y

    def get_model(self, args, adj):
        if args.model_name == "GraphGRU": # GCN + GRU
            net = models.GraphGRU(adj=adj, hidden_dim=args.hidden_num, pred_len=args.pred_len, num_features=1)
        elif args.model_name == "DMA_STGCRNN":# multi-attribute GCN + GRU
            net = models.DMA_STGCRNN(adj=adj, hidden_dim=args.hidden_num, num_features=5,
                              dropout_probability=args.dropout_probability,
                              pred_len=args.pred_len)
        else:
            raise Exception('invalid model name')
        return net


# train network using multi-GPUs
def train_network_ddp(model, trainloader, args, criterion, optimizer, scheduler, local_rank, world_size):
    logfile = open(args.log_dir + '/' + args.model_name + 'loss.txt', 'w')
    loss_list = []
    model.train()
    for epoch in range(args.epoch):
        epoch_loss_list = []
        for step, (batch_x, batch_y) in enumerate(trainloader):
            batch_x = batch_x.to(local_rank)
            batch_y = batch_y.to(local_rank)
            prediction = model(batch_x)
            loss = criterion(prediction[:, list(range(args.pred_len)), :], batch_y[:, list(range(args.pred_len)), :])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            # nn.utils.clip_grad_value_(model.regressor.weight, clip_value=100)
            optimizer.step()

            # print loss
            n_gpus = torch.cuda.device_count()
            dist.reduce(loss, dst=0)
            if local_rank == 0:
                if step % 100 == 0:
                    print("Epoch: ", epoch, "| Step: ", step,
                          "|loss: ", loss.item() / n_gpus)
                    print("Epoch: ", epoch, "| Step: ", step,
                          "|loss: ", loss.item() / n_gpus, file=logfile)
            epoch_loss_list.append(loss.item() / n_gpus)
        scheduler.step()
        # 打印当前学习率
        if local_rank == 0:
            mean_loss = np.mean(epoch_loss_list)
            print("Epoch: ", epoch, "|  Mean Loss:", mean_loss)
            print("Epoch: ", epoch, "|  Mean Loss:", mean_loss, file=logfile)
            current_lr = scheduler.get_last_lr()[0]
            print(f"epoch {epoch}: Current learning rate = {current_lr}")
            print(f"epoch {epoch}: Current learning rate = {current_lr}", file=logfile)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, args.log_dir + '/' + args.model_name + '_v1_epoch' + str(epoch) + '.pth')


# train network using single GPU
def train_network_single_gpu(model, trainloader, args, criterion, optimizer, scheduler):
    logfile = open(args.log_dir + '/' + args.model_name + 'loss.txt', 'w')
    for epoch in range(args.epoch):
        epoch_loss_list = []
        for step, (batch_x, batch_y) in enumerate(trainloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x)
            loss = criterion(prediction[:, list(range(args.pred_len)), :], batch_y[:, list(range(args.pred_len)), :])
            # print loss
            if step % 100 == 0:
                print("Epoch: ", epoch, "| Step: ", step,
                      "|loss: ", loss.item())
                print("Epoch: ", epoch, "| Step: ", step,
                      "|loss: ", loss.item(), file=logfile)
            epoch_loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
            optimizer.step()
        scheduler.step()
        mean_loss = np.mean(epoch_loss_list)
        print("Epoch: ", epoch, "|  Mean Loss:", mean_loss)
        print("Epoch: ", epoch, "|  Mean Loss:", mean_loss, file=logfile)
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}: Current learning rate = {current_lr}")
        print(f"epoch {epoch}: Current learning rate = {current_lr}", file=logfile)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, args.log_dir + '/' + args.model_name + '_v1_epoch' + str(epoch) + '.pth')

if __name__ == "__main__":
    data_paths = {
        "pvpower": {"feat": "data/data.mat", 
                    "adj": "data/pv_multi_adj.npy"}
    }
    # define network hyperparameters
    args = default_hyperparameters()

    # create log folder to save results
    os.makedirs('./' + args.log_dir, exist_ok=True)

    if args.ddp_training == 'True':
        # init
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)

        # load data
        if args.model_name == "GraphGRU":
            trainloader, trainloader1, testloader, adj_matrix = PV_single_dataloader_func(args)
        else:
            trainloader, trainloader1, testloader, adj_matrix = PV_dataloader_func(args)

        # establish and train network
        model = Network(args, adj_matrix)
        model.to(local_rank)
        criterion = nn.MSELoss().to(local_rank)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lrDecay)
        train_network_ddp(model, trainloader, args, criterion, optimizer, scheduler, local_rank, world_size)
    elif args.ddp_training == 'False':
        # load data
        xtrain, ytrain, xtest, ytest, adj_matrix, \
            xtrainnor, ytrainnor, xtestnor, ytestnor, \
            xtrainmin, xtrainmax, ytrainmin, ytrainmax = load_data(args, data_paths)
        trainnor_x = torch.from_numpy(xtrainnor).type(torch.float32)
        trainnor_y = torch.from_numpy(ytrainnor).type(torch.float32)
        testnor_x = torch.from_numpy(xtestnor).type(torch.float32)
        testnor_y = torch.from_numpy(ytestnor).type(torch.float32)
        if args.model_name == "GraphGRU":
            xtrainnor = xtrainnor[:, :, :, 0]
            trainnor_x = torch.from_numpy(xtrainnor.reshape(xtrainnor.shape[0],
                                                            xtrainnor.shape[1],
                                                            xtrainnor.shape[2], 1)).type(torch.float32)
            xtestnor = xtestnor[:, :, :, 0]
            testnor_x = torch.from_numpy(xtestnor.reshape(xtestnor.shape[0],
                                                        xtestnor.shape[1],
                                                        xtestnor.shape[2], 1)).type(torch.float32)
        train_torch_dataset = torch.utils.data.TensorDataset(trainnor_x, trainnor_y)
        trainloader = torch.utils.data.DataLoader(
            dataset=train_torch_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0)
        test_torch_dataset = torch.utils.data.TensorDataset(testnor_x, testnor_y)
        testloader = torch.utils.data.DataLoader(
            dataset=test_torch_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0)
        # model
        model = Network(args, adj_matrix)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate,
                                     weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.lrDecay)
        train_network_single_gpu(model, trainloader, args, criterion, optimizer, scheduler)
    else:
        raise ValueError("Invalid value for --ddp_training. Use 'True' or 'False'.")