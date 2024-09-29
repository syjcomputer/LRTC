import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
datasetPath = os.path.join(os.path.split(rootPath)[0], 'datasets')
sys.path.append(rootPath)

"""
datasets selection
0: bird-40
"""

global_data_selection = 5
global_deep_learning_selection = 0
global_risk_dataset = 'risk_dataset'

class Configuration(object):
    def __init__(self, data_selection, deep_learning_selection, risk_dataset = 'risk_dataset'):
        self.data_selection = data_selection
        self.deep_learning_selection = deep_learning_selection

        # setting risk dataset
        # 2020.07.11 remove the datasets file to parent directory
        self.data_dict = {
            0: os.path.join(datasetPath, 'bird-10'),
            1: os.path.join(datasetPath, 'bird-40'),
            2: os.path.join(datasetPath, 'bird-40-10'),
            3: os.path.join(datasetPath, 'bird-40-30-10'),
            4: os.path.join(datasetPath, 'bird-40-20-10'),

            # xubo add
            5: r'G:\ICRA\tfidf_test'
        }
        self.class_num_dict = {
            0: 10,
            1: 40,
            2: 40,
            3: 40,
            4: 40,

            # xubo add
            5: 11
        }

        # image_dataset: image dataset divide
        # risk_dataset: the dataset for risk model, which includes all_data_info.csv, train.csv,
        #             val.csv, test.csv, decision_tree_rules_clean.txt, decision_tree_rules_info
        # npy_dataset: the dataset from interpretable neural network
        # dataset2csv: extract the risk csv of each class from the npy_dataset, which uses to generate the rules of each class
        # dataset2mulcsv: extract all risk csv of all class from the bpy_dataset, which uses to merge the risk_dataset
        # rules: save the rules of each class, which uses to merge all the rules for risk dataset
        # base_risk_nums: how many prototypes are there for each class

        self.data_path = self.data_dict[self.data_selection]
        self.image_dataset_path = os.path.join(self.data_path, 'image_dataset')
        self.risk_dataset_path = os.path.join(self.data_path, 'risk_dataset')
        self.npy_dataset_path = os.path.join(self.data_path, 'npy_dataset')
        self.data2csv_path = os.path.join(self.data_path, 'data2csv')
        self.data2mulcsv_path = os.path.join(self.data_path, 'data2mulcsv')
        self.rules_path = os.path.join(self.data_path, 'rules')
        self.base_risk_nums = 10
        self.base_risk_list = ['den201', 'res34', 'vgg19']


        # setting epochs
        # these parameters are not used now
        self.train_size = 20
        self.deep_learning_epochs = 1
        self.risk_epochs = 100

        # setting risker
        # 2020.7.11 change minimum_observation_num to percentage
        #
        # minimum_observation_num: the rule matchs minimun of data
        # rule_acc: the lowest accuracy of this rule on the validation

        self.interval_number_4_continuous_value = 100
        self.learing_rate = 0.001
        self.risk_training_epochs = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.minimum_observation_num = 0
        self.rule_acc = 0.
        self.risk_confidence = 0.90
        self.model_save_path = os.path.join(self.risk_dataset_path, 'tf_model')


        # setting decision_tree
        self.match_gini = 0.01
        self.unmatch_gini = 0.01
        self.tree_depth = 1
        self.generate_rules = False

        self.raw_data_path = None
        self.raw_decision_tree_rules_path = None
        self.decision_tree_rules_path = os.path.join(self.risk_dataset_path, 'decision_tree_rules_clean.txt')
        self.info_decision_tree_rules_path = os.path.join(self.risk_dataset_path, 'decision_tree_rules_info.txt')
        self.train = None

        # the frame is used
        self.risk_model_type = 'f'  # torch or tf

    def get_parent_path(self):
        return self.risk_dataset_path

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
        return os.path.join(self.risk_dataset_path, 'all_data_info.csv')
