import numpy as np
import pandas as pd
import torch
import scipy.io

# load data
def load_data(args, data_paths):
    data_path = data_paths[args.data]["feat"]
    adj_path = data_paths[args.data]["adj"]
    seq_len = args.seq_len
    if args.data == "pvpower":
        data = scipy.io.loadmat(data_path)
        pvtrainall = data['pvtrainall']
        pvtestall = data['pvtestall']
        ghitrainall = data['ghitrainall']
        ghitestall = data['ghitestall']
        temtrainall = data['temtrainall']
        temtestall = data['temtestall']
        cleartrainall = data['clearghi_trainall']
        cleartestall = data['clearghi_testall']
        traindata = np.concatenate((pvtrainall.reshape(-1, pvtrainall.shape[1], 1),
                                    ghitrainall.reshape(-1, pvtrainall.shape[1], 1),
                                    temtrainall.reshape(-1, pvtrainall.shape[1], 1),
                                    cleartrainall.reshape(-1, pvtrainall.shape[1], 1)), axis=2)
        testdata = np.concatenate((pvtestall.reshape(-1, pvtrainall.shape[1], 1),
                                   ghitestall.reshape(-1, pvtrainall.shape[1], 1),
                                   temtestall.reshape(-1, pvtrainall.shape[1], 1),
                                   cleartestall.reshape(-1, pvtrainall.shape[1], 1)), axis=2)
        '''
        data = np.load('./data/pvnwp_data_normalized.npy')
        traindata = data[:105216, :]  # 2016-2018, each day has 96 datapoints
        testdata = data[105216:, :]  # 2019-2020
        '''

        adj_matrix = np.load(adj_path)
    else:
        raise Exception("invlaid dataset name")

    xtrain, ytrain, xtest, ytest = [], [], [], []
    for i in range(traindata.shape[0] - seq_len - seq_len):
        a = traindata[i: i + seq_len + seq_len, :, :]
        cleara = a[seq_len: seq_len + seq_len, :, -1].reshape(-1, traindata.shape[1], 1)  # clear-sky ghi
        xtrain.append(np.concatenate((a[0: seq_len, :, :], cleara), axis=2))
        ytrain.append(a[seq_len: seq_len + seq_len, :, 0])
    for i in range(testdata.shape[0] - seq_len - seq_len):
        b = testdata[i: i + seq_len + seq_len, :, :]
        clearb = b[seq_len: seq_len + seq_len, :, -1].reshape(-1, traindata.shape[1], 1)   # clear-sky ghi
        xtest.append(np.concatenate((b[0: seq_len, :, :], clearb), axis=2))
        ytest.append(b[seq_len: seq_len + seq_len, :, 0])

    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtest = np.array(xtest)
    ytest = np.array(ytest)

    # obtain the min and max to normalize the data by (x-xmin)/(xmax-xmin)
    xtrainmin = np.min(xtrain, axis=0)
    xtrainmax = np.max(xtrain, axis=0)
    ytrainmin = np.min(ytrain, axis=0)
    ytrainmax = np.max(ytrain, axis=0)
    xtrainnor = (xtrain - xtrainmin) / (xtrainmax - xtrainmin)
    xtestnor = (xtest - xtrainmin) / (xtrainmax - xtrainmin)
    ytrainnor = ytrain  # (ytrain - ytrainmin) / (ytrainmax - ytrainmin)
    ytestnor = ytest  # (ytest - ytrainmin) / (ytrainmax - ytrainmin)

    return xtrain, ytrain, xtest, ytest, adj_matrix, \
        xtrainnor, ytrainnor, xtestnor, ytestnor, \
        xtrainmin, xtrainmax, ytrainmin, ytrainmax

def PV_dataloader_func(args):
    # path for data and adj_matrix
    data_paths = {
        "pvpower": {"feat": "data/data.mat", "adj": "data/pv_multi_adj.npy"}
    }

    # load data
    xtrain, ytrain, xtest, ytest, adj_matrix, \
        xtrainnor, ytrainnor, xtestnor, ytestnor, \
        xtrainmin, xtrainmax, ytrainmin, ytrainmax = load_data(args, data_paths)
    trainnor_x = torch.from_numpy(xtrainnor).type(torch.float32)
    trainnor_y = torch.from_numpy(ytrainnor).type(torch.float32)
    testnor_x = torch.from_numpy(xtestnor).type(torch.float32)
    testnor_y = torch.from_numpy(ytestnor).type(torch.float32)
    
    train_torch_dataset = torch.utils.data.TensorDataset(trainnor_x, trainnor_y)  # tuple (x, y)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_torch_dataset)
    trainloader = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler)  # for training    
    trainloader1 = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=10240,
        shuffle=False,
        num_workers=0)  # for evaluating training results    
    test_torch_dataset = torch.utils.data.TensorDataset(testnor_x, testnor_y)  # tuple (x, y)
    testloader = torch.utils.data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=10240,
        shuffle=False,
        num_workers=0)  # for evaluating test results
    return trainloader, trainloader1, testloader, adj_matrix

def PV_single_dataloader_func(args):
    # path for data and adj_matrix
    data_paths = {
        "pvpower": {"feat": "data/data.mat", "adj": "data/pv_multi_adj.npy"}
    }
    # load data
    xtrain, ytrain, xtest, ytest, adj_matrix, \
        xtrainnor, ytrainnor, xtestnor, ytestnor, \
        xtrainmin, xtrainmax, ytrainmin, ytrainmax = load_data(args, data_paths)
    xtrainnor = xtrainnor[:, :, :, 0]
    trainnor_x = torch.from_numpy(xtrainnor.reshape(xtrainnor.shape[0], 
                                                  xtrainnor.shape[1],
                                                  xtrainnor.shape[2], 1)).type(torch.float32)
    trainnor_y = torch.from_numpy(ytrainnor).type(torch.float32)
    xtestnor = xtestnor[:, :, :, 0]
    testnor_x = torch.from_numpy(xtestnor.reshape(xtestnor.shape[0], 
                                                  xtestnor.shape[1],
                                                  xtestnor.shape[2], 1)).type(torch.float32)
    testnor_y = torch.from_numpy(ytestnor).type(torch.float32)
    
    train_torch_dataset = torch.utils.data.TensorDataset(trainnor_x, trainnor_y)  # tuple (x, y)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_torch_dataset)
    trainloader = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler)  # for training    
    trainloader1 = torch.utils.data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=10240,
        shuffle=False,
        num_workers=0)  # for evaluating training results    
    test_torch_dataset = torch.utils.data.TensorDataset(testnor_x, testnor_y)  # tuple (x, y)
    testloader = torch.utils.data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=10240,
        shuffle=False,
        num_workers=0)  # for evaluating test results
    return trainloader, trainloader1, testloader, adj_matrix