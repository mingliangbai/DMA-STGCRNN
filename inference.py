'''
# inference for DMA_STGCRNN (multi-attribute GCN + GRU)
python  inference.py --model_name DMA_STGCRNN --log_dir log --pred_len 16 --hidden_num 240

# inference for GraphGRU (GCN + GRU)
python  inference.py --model_name GraphGRU --log_dir log --pred_len 16 --hidden_num 240
'''
import models
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from config.GraphGRU_config import default_hyperparameters
import scipy.io
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

# network inference
def inference(model, trainloader1, testloader, device, args, McDropout=False):
    if McDropout:
        model.train()
    else:
        model.eval()
    with torch.no_grad():
        ytrainprenor = torch.empty((0, args.pred_len, 11))  # (batch_size, pred_len, num_nodes), the batch dim is empty
        for step, (batch_x, batch_y) in enumerate(trainloader1):
            batch_x = batch_x.to(device)
            prediction = model(batch_x)
            ytrainprenor = torch.cat((ytrainprenor, prediction.cpu()), dim=0)
        ytestprenor = torch.empty((0, args.pred_len, 11))  # (batch_size, pred_len, num_nodes), the batch dim is empty
        for step, (batch_x, batch_y) in enumerate(testloader):
            batch_x = batch_x.to(device)
            prediction = model(batch_x)
            ytestprenor = torch.cat((ytestprenor, prediction.cpu()), dim=0)
    ytrainprenor = ytrainprenor.numpy()
    ytestprenor = ytestprenor.numpy()
    return ytrainprenor, ytestprenor


def inverse_normalize(predictionnor, ytrainmin, ytrainmax):
    actualvalue = predictionnor * (ytrainmax - ytrainmin) + ytrainmin
    return actualvalue


def evaluation(ytrainpre, ytrain, ytestpre, ytest):
    # training set
    trainmae = np.mean(np.abs(ytrainpre - ytrain))
    trainmae_detail = np.mean(np.abs(ytrainpre - ytrain), axis=0)
    trainrmse = np.sqrt(np.mean((ytrainpre - ytrain) ** 2))
    trainrmse_detail = np.sqrt(np.mean((ytrainpre - ytrain) ** 2, axis=0))

    # test set
    testmae = np.mean(np.abs(ytestpre - ytest))
    testmae_detail = np.mean(np.abs(ytestpre - ytest), axis=0)
    testrmse = np.sqrt(np.mean((ytestpre - ytest) ** 2))
    testrmse_detail = np.sqrt(np.mean((ytestpre - ytest) ** 2, axis=0))
    return trainmae, trainmae_detail, trainrmse, trainrmse_detail, \
        testmae, testmae_detail, testrmse, testrmse_detail


if __name__ == "__main__":
    # define network hyperparameters
    args = default_hyperparameters()

    # make directory to save results
    os.makedirs(args.log_dir +'/result', exist_ok=True)
  
    
    # load data
    data_paths = {
        "pvpower": {"feat": "data/data.mat", "adj": "data/pv_multi_adj.npy"}
    }
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
        num_workers=0)  # for training
    test_torch_dataset = torch.utils.data.TensorDataset(testnor_x, testnor_y) 
    testloader = torch.utils.data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)  # for inference
    
    # inference with one GPU
    model = Network(args, adj_matrix)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    if args.model_name == "GraphGRU":
        checkpoint = torch.load(args.log_dir + '/GraphGRU_v1_epoch1.pth')
    elif args.model_name == "DMA_STGCRNN":
        checkpoint = torch.load(args.log_dir + '/DMA_STGCRNN_v1_epoch1.pth')
    else:
        raise Exception('invalid model name')
    model.load_state_dict(checkpoint['model_state_dict'])
    ytrainprenor, ytestprenor = inference(model, trainloader, testloader, device, args, McDropout=False)
    ytrainpre = ytrainprenor  # inverse_normalize(ytrainprenor, ytrainmin, ytrainmax)
    ytestpre = ytestprenor  # inverse_normalize(ytestprenor, ytrainmin, ytrainmax)

    # evaluation
    trainmae, trainmae_detail, trainrmse, trainrmse_detail, \
        testmae, testmae_detail, testrmse, testrmse_detail = evaluation(ytrainpre,
                                                                        ytrain[:, list(range(args.pred_len)), :],
                                                                        ytestpre,
                                                                        ytest[:, list(range(args.pred_len)), :])
    logfile = open(args.log_dir + '/result/' + args.model_name + 'result_epoch.txt', 'w')
    print('trainmae:', trainmae, file=logfile)
    print('trainmae_step:', trainmae_detail, file=logfile)
    print('trainrmse:', trainrmse, file=logfile)
    print('trainrmse_detail:', trainrmse_detail, file=logfile)
    print('testmae:', testmae, file=logfile)
    print('testmae_step:', testmae_detail, file=logfile)
    print('testrmse:', testrmse, file=logfile)
    print('testrmse_detail:', testrmse_detail, file=logfile)
    np.save(args.log_dir +'/result/ytrainpre.npy', ytrainpre)
    np.save(args.log_dir +'/result/ytestpre.npy', ytestpre)
    np.save(args.log_dir +'/result/ytrain.npy', ytrain)
    np.save(args.log_dir +'/result/ytest.npy', ytest)
    np.savetxt(args.log_dir +'/result/testrmse_detail.txt', testrmse_detail)
    np.savetxt(args.log_dir + '/result/testmae_detail.txt', testmae_detail)



