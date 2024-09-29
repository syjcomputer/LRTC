import argparse
from PrepareRiskDataset.config.overall_config import *

# 基本信息
num_cores = 10
dataset = source_dataset # train dataset
data_name = f'{dataset}_bert'
cur_path = './{}/{}/'.format(data_name, rate)
model_name = "textcnn"

if model_name=="bert":
    maxlen = 150
    epochs = 1
    batch_size = 64
    learning_rate = 4e-5  # 4e-5

elif model_name=="textcnn":

    def get_params():
        # Training settings
        parser = argparse.ArgumentParser(description='train textcnn')
        # need revise
        parser.add_argument('--label_num', type=int, default=classify_num_labels,
                            help='blank')
        parser.add_argument('--filter_num', type=int, default=16,
                            help='blank')
        parser.add_argument('--filter_sizes', type=list, default=[2, 3, 4],
                            help='blank')
        parser.add_argument('--vocab_size', type=int, default=3,
                            help='blank')
        parser.add_argument('--embedding_dim', type=int, default=100,
                            help='blank')
        parser.add_argument('--static', type=bool, default=True,
                            help='blank')
        parser.add_argument('--dropout', type=float, default=0.7,
                            help='blank')

        parser.add_argument('--vectors', type=torch.Tensor, default=None,
                            help='blank')
        parser.add_argument('--data_name', type=str, default=dataset,
                            help="train dataset")
        parser.add_argument('--rate', type=float, default=1.0,
                            help='training rate')

        # parameters can be tuning
        parser.add_argument('--seed', type=int, default=1234,
                            help='random seed of settings')
        parser.add_argument('--epoches', type=int, default=1,
                            help='epoch (RiskModel.train) numbers')
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='learing rate of RiskTorchModel')
        parser.add_argument('--bs', type=int, default=64,
                            help='batch size in RiskTorchModel')

        args, _ = parser.parse_known_args()
        return args