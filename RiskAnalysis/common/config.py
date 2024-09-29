import sys
import os
import argparse

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
datasetPath = os.path.join(os.path.split(rootPath)[0], 'datasets')
sys.path.append(rootPath)

config_path = 'uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

global_data_selection = 27
global_data = "AgNews"
global_risk_dataset = '1.0'

global_finetune_data = 23
global_finetune = "BBC"
global_finetune_risk_dataset = '1.0'

base_model_name = 'best_model.weights'

class Configuration(object):
    def __init__(self, data_selection, deep_learning_selection, risk_dataset):
        self.data_selection = data_selection
        self.deep_learning_selection = deep_learning_selection

        # setting risk dataset
        self.data_dict = {
            18: './datasets/fudan',
            19: 'datasets/20News',
            20: './datasets/NYT',
            21: './datasets/qinghua',
            22: './datasets/jiaocha',
            23: 'datasets/BBC',
            24: 'datasets/SST2',
            25: 'datasets/IMDB',
            26: "datasets/MR",
            27: os.path.join(os.path.split(rootPath)[0], 'PrepareRiskDataset/AgNews')
        }
        self.class_num_dict = {
            18: 20,
            19: 3, #20,
            20: 26,
            21: 14,
            22: 3,
            23: 4,
            24: 2,
            25: 2,
            26:2,
            27:4,
        }

        self.data_name = self.data_dict[self.data_selection].split('/')[-1]
        self.data_path = self.data_dict[self.data_selection]    # train data
        # self.risk_dataset_path = os.path.join(self.data_path, risk_dataset)
        self.risk_dataset_path = os.path.join(os.path.split(rootPath)[0], f'PrepareRiskDataset/{global_data}')
        self.npy_dataset_path = os.path.join(self.data_path, 'npy_dataset')
        self.data2csv_path = os.path.join(self.data_path, 'data2csv')
        self.data2mulcsv_path = os.path.join(self.data_path, 'data2mulcsv')
        self.rules_path = os.path.join(self.data_path, 'decision_tree_rules_clean.txt')
        # self.base_risk_nums = 10
        # self.base_risk_list = ['den169', 'res18', 'res34']

        self.interval_number_4_continuous_value = 50
        self.learing_rate = 0.001 # 0.001
        self.risk_training_epochs = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.minimum_observation_num = 0.0  # 规则支持度
        self.rule_acc = 0.0  # 0.0 根据准确率筛选规则
        self.risk_confidence = 0.90
        self.model_save_path = os.path.join(self.risk_dataset_path, 'tf_model')

        # setting decision_tree
        self.match_gini = 0.2
        self.unmatch_gini = 0.00001
        self.tree_depth = 1  # 1
        self.generate_rules = False

        self.raw_data_path = None
        self.raw_decision_tree_rules_path = os.path.join(os.path.split(rootPath)[0], f'OneSideDecisionTree/{global_data}_output/1.0')
        self.decision_tree_rules_path = os.path.join(self.raw_decision_tree_rules_path, 'decision_tree_rules_clean.txt')
        self.info_decision_tree_rules_path = os.path.join(self.raw_decision_tree_rules_path, 'decision_tree_rules_info.txt')
        self.train = None

        # the frame is used
        self.risk_model_type = 'f'  # torch or tf

        # fine-tune settings
        self.fine_tune = True
        self.fine_tune_data =  self.data_dict[global_data_selection]
        self.fine_tune_dataset = global_risk_dataset
        self.fine_tune_path = os.path.join(f'{self.fine_tune_data}_bert', self.fine_tune_dataset)

    def get_finetune_path(self):
        # corresponding to get_parent_path and get_risk_dataset_path
        return self.fine_tune_path

    def get_parent_path(self):
        return f"{self.risk_dataset_path}_cur/1.0"

    def get_npy_dataset_path(self):
        return self.npy_dataset_path

    def get_data2csv_path(self):
        return self.data2csv_path

    def get_data2mulcsv_path(self):
        return self.data2mulcsv_path

    def get_risk_dataset_path(self):
        return self.risk_dataset_path

    def get_class_num(self):
        return self.class_num_dict[self.data_selection]

    def get_rules_dataset_path(self):
        return self.rules_path

    def get_raw_decision_tree_rules_path(self):
        return self.raw_decision_tree_rules_path

    def get_info_decision_tree_rules_path(self):
        return self.info_decision_tree_rules_path

    def get_decision_tree_rules_path(self):
        return self.decision_tree_rules_path

    def get_raw_data_path(self):
        return self.raw_data_path

    def get_train_path(self):
        return self.train

    def get_all_data_path(self):
        # print(self.risk_dataset_path)
        return os.path.join(self.get_parent_path(), 'all_data_info.csv')

    def get_params(self):
        # Training settings
        parser = argparse.ArgumentParser(description='ada_train')

        # almost fixed parameters
        parser.add_argument('--device', type=int, default=0,
                            help='0 represents CUDA, else cpu')
        parser.add_argument('--num_cores', type=int, default=10,
                            help='intra_op_parallelism_threads in tensorflow')
        parser.add_argument('--learn_confidence', type=float, default=0.9,
                            help='learn confidence of risk model')
        parser.add_argument('--class_num', type=int, default=self.class_num_dict[global_data_selection],
                            help='blank')
        parser.add_argument('--store_name', type=str, default=global_data,
                            help='save the new base model')
        parser.add_argument('--resume', type=bool, default=True,
                            help='resume training from checkpoint')
        parser.add_argument('--data_selection', type=int, default=global_data_selection,
                            help='train data selection in config.py')
        parser.add_argument('--deep_learning_selection', type=int, default=0,
                            help='config.py, fine-tune data')
        parser.add_argument('--maxlen', type=int, default=150,
                            help='train()')
        parser.add_argument('--rate', type=float, default=1.0,
                            help='train rate')
        parser.add_argument("--model_path2", type=str,
                            default=fr"{os.path.split(rootPath)[0]}/PrepareRiskDataset/{global_data}_bert/1.0/{base_model_name}",
                            help="trained baseline model for bert4keras")

        # parameters can be tuning
        parser.add_argument('--seed', type=int, default=1234,
                            help='random seed of settings')
        parser.add_argument('--nb_epoch', type=int, default=3,
                            help='ending epoch of adaptive model')
        parser.add_argument('--epoches', type=int, default=5,
                            help='epoch (RiskModel.train) numbers')
        parser.add_argument('--learning_rate', type=float, default=5e-5,  # 20News_BBC参数5e-5
                            help='learing rate of RiskTorchModel')
        parser.add_argument('--bs', type=int, default=64,  # 16
                            help='batch size in RiskTorchModel')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size in train(), risk model dataset')
        parser.add_argument("--lr2", type=float, default=5e-5,  # 20News_BBC参数5e-5
                            help="adaptive model lr")

        args, _ = parser.parse_known_args()
        return args

cfg = Configuration(global_data_selection, 0, global_risk_dataset)