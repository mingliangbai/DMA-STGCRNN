import argparse


def default_hyperparameters():
    parser = argparse.ArgumentParser(description='Hyperparameters.')
    parser.add_argument('--model_name', type=str, help='the name of the model',
                        choices={'DMA_STGCRNN','GraphGRU'},
                        default='DMA_STGCRNN'
                        )
    parser.add_argument('--data', type=str, help='the name of the dataset',
                        choices={'pvpower'},
                        default='pvpower'
                        )
    parser.add_argument('--seq_len', type=int, help='number of used historical datapoints to predict the future',
                        default=96
                        )
    parser.add_argument('--pred_len', type=int, help='prediction steps, namely the output dimension of network',
                        default=16
                        )
    parser.add_argument('--hidden_num', type=int, help='number of hidden neurons',
                        default=480
                        )
    parser.add_argument('--batch_size', type=int, help='the batch size',
                        default=256
                        )
    parser.add_argument('--learning_rate', type=float, help='learning rate',
                        default=0.001
                        )
    parser.add_argument('--weight_decay', type=float, help='weight_decay or l2 regularization coefficient',
                        default=0)
    parser.add_argument('--dropout_probability', type=float, help='dropout_probability',
                        default=0.1)
    parser.add_argument('--epoch', type=int, help='number of epochs for training',
                        default=500)  # 500
    parser.add_argument('--log_dir', type=str, help='Path to the output console log file',
                        default='log'
                        )
    parser.add_argument('--lrDecay', type=float, help='learning rate decay every 10 epoch',
                        default=0.95
                        )
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank of the process')
    parser.add_argument('--ddp_training', type=str, help='whether to use ddp multi-GPU training or not',
                        choices={'True', 'False'},
                        default='False'
                        )
    args = parser.parse_args()
    return args
