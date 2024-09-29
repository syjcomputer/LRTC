import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

"""
Data sets selection:
0: Abt-Buy
1: DBLP-Scholar
2: songs
3: Amazon-Google
4: restaurant
5: products
6: DBLP-ACM
7: Citeseer-DBLP
8: Data shifting: Abt-Buy to Amazon-Google
9: Data shifting: DBLP-Scholar to DBLP-ACM
10: Data shifting: DBLP-ACM to DBLP-Scholar
"""

global_selection = 1


class Configuration(object):
    def __init__(self, selection):
        self.data_selection = selection
        self.train_valida_test_ratio = {0: '127', 1: '226', 2: '325', 3: '3valida400', 4: '3valida1k'}
        self.path_dict = {0: rootPath + '/input_data/Abt-Buy/',
                          1: rootPath + '/input_data/DBLP-Scholar/',
                          2: rootPath + '/input_data/songs/',
                          3: rootPath + '/input_data/Amazon-Google/',
                          4: rootPath + '/input_data/restaurant/',
                          5: rootPath + '/input_data/products/',
                          6: rootPath + '/input_data/DBLP-ACM/',
                          7: rootPath + '/input_data/Citeseer-DBLP/',
                          8: rootPath + '/input_data/AB2AG/',
                          9: rootPath + '/input_data/DS2DA/',
                          10: rootPath + '/input_data/DA2DS/'}

        self.data_source1_dict = {0: self.path_dict.get(0) + 'Abt.csv',
                                  1: self.path_dict.get(1) + 'DBLP1.csv',
                                  2: self.path_dict.get(2) + 'msd.csv',
                                  3: self.path_dict.get(3) + 'Amazon.csv',
                                  4: self.path_dict.get(4) + 'fodors.csv',
                                  5: self.path_dict.get(5) + 'new_walmart.csv',
                                  6: self.path_dict.get(6) + 'DBLP2.csv',
                                  7: self.path_dict.get(7) + 'new_citeseer.csv',
                                  8: self.path_dict.get(8) + 'Amazon.csv',
                                  9: self.path_dict.get(9) + 'DBLP2.csv',
                                  10: self.path_dict.get(10) + 'DBLP1.csv'}

        self.data_source2_dict = {0: self.path_dict.get(0) + 'Buy_new.csv',
                                  1: self.path_dict.get(1) + 'Scholar.csv',
                                  2: None,
                                  3: self.path_dict.get(3) + 'GoogleProducts.csv',
                                  4: self.path_dict.get(4) + 'zagats.csv',
                                  5: self.path_dict.get(5) + 'new_amazon.csv',
                                  6: self.path_dict.get(6) + 'ACM.csv',
                                  7: self.path_dict.get(7) + 'new_dblp.csv',
                                  8: self.path_dict.get(8) + 'GoogleProducts.csv',
                                  9: self.path_dict.get(9) + 'ACM.csv',
                                  10: self.path_dict.get(10) + 'Scholar.csv'}

        self.tvt_selection = 2
        self.risk_training_size = 20
        self.random_select_risk_training = False
        self.risk_confidence = 0.9
        self.minimum_observation_num = 5
        self.budget_levels = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                              1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                              2500, 3000, 3500, 4000, 4500, 5000]
        self.interval_number_4_continuous_value = 50
        self.learn_variance = True
        self.apply_function_to_weight_classifier_output = True
        self.deepmatcher_epochs = 10
        self.risk_training_epochs = 1000

    def get_parent_path(self):
        return self.path_dict.get(self.data_selection) + self.train_valida_test_ratio.get(self.tvt_selection) + '/'

    def get_data_source_1(self):
        return self.data_source1_dict.get(self.data_selection)

    def get_data_source_2(self):
        return self.data_source2_dict.get(self.data_selection)

    def get_raw_data_path(self):
        return self.path_dict.get(self.data_selection) + 'pair_info_more.csv'

    def get_shift_raw_data_path(self):
        return self.path_dict.get(self.data_selection) + 'pair_info_more_2.csv'

    def get_raw_decision_tree_rules_path(self):
        return self.get_parent_path() + 'decision_tree_rules_raw.txt'

    def get_decision_tree_rules_path(self):
        return self.get_parent_path() + 'decision_tree_rules_clean.txt'

    def use_other_domain_workload(self):
        return self.data_selection in {8, 9, 10}

    @staticmethod
    def get_budgets():
        budget_levels = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                         1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                         2500, 3000, 3500, 4000, 4500, 5000]
        # budget_levels = [i for i in range(30, 150, 30)]
        # budget_levels = [i for i in range(500, 20500, 500)]
        return budget_levels
