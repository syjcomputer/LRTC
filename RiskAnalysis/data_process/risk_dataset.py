import numpy as np
from scipy import sparse as sp
from RiskAnalysis.data_process import similarity_based_feature as sbf
from RiskAnalysis.common import utils
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import logging
import time
import pandas as pd
from RiskAnalysis.data_process import rules_process as rp
from tqdm import tqdm
from RiskAnalysis.common.config import cfg, global_risk_dataset
import math
import os

# cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection,
#                            config.global_risk_dataset)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")


class DataInfo(object):
    def __init__(self, ids, id_2_pair_info, name=''):
        # Ground truth info.
        #self.data_name = dataset
        self.data_name = name
        #self.data_name = name
        self.data_ids = ids
        self.data_len = len(ids)
        self.true_labels = None
        self.id_2_true_labels = None
        self.update_ground_truth_info(id_2_pair_info)
        # Machine info.
        self.machine_probs = None
        self.machine_mul_probs = None
        self.id_2_probs = None
        self.id_2_mul_probs = None
        self.machine_labels = None
        self.machine_label_2_one = None
        self.id_2_machine_labels = None
        self.risk_labels = None
        self.risk_mul_labels = None
        self.risk_activate = None
        self.class_num = cfg.get_class_num()
        # rule-based features
        self.original_rule_activation_matrix = None
        self.rule_activation_matrix = None
        self.risk_mean_X_discrete = None
        self.risk_variance_X_discrete = None
        # probability-based features
        self.prob_activation_matrix = None
        self.risk_mean_X_continue = None
        self.risk_variance_X_continue = None
        # distribution info of rules
        self.mu_vector = None
        self.sigma_vector = None
        self.rules = None
        self.rules_idx = None
        self.rules_max = 0
        # risk information of the data
        self.risk_values = None
        self.pair_mus = None
        self.pair_sigmas = None
        # Select partial data as activated.
        self.activate_data_ids = None
        self.activate_data_idx = None
        self.id_2_index = dict()
        for _idx, _id in enumerate(self.data_ids):
            self.id_2_index[_id] = _idx

    def update_ground_truth_info(self, id_2_pair_info):
        self.true_labels = []
        self.id_2_true_labels = dict()
        for _id in self.data_ids:
            ground_truth_label = id_2_pair_info.get(_id)[1]
            self.true_labels.append(ground_truth_label)
            self.id_2_true_labels[_id] = ground_truth_label
        self.true_labels = np.array(self.true_labels)
        path = f"./results/{cfg.data_name}/{global_risk_dataset}"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '/' + self.data_name + '_true_label.npy', self.true_labels)
        #np.save('./results/fudan/' + self.data_name + '_true_label.npy', self.true_labels)

    def update_machine_mul_info(self, machine_mul_probs):
        '''
        for i in range(self.data_len):
            for j in range(self.class_num):
                machine_mul_probs[i][j] = np.log(machine_mul_probs[i][j] + 1) / np.log(2)
        '''
        self.machine_mul_probs = machine_mul_probs

        self.id_2_mul_probs = dict()
        for i in range(self.data_len):
            self.id_2_mul_probs[self.data_ids[i]] = [self.machine_mul_probs[i]]

    def update_machine_info(self, machine_probs, machine_labels=None, activate_ids=None):

        """
        Note: If the activate_ids is not None,
                the input machine_probs and activate_ids should share a one-to-one mapping.
              Otherwise, the machine_probs is treated having the same order as the self.data_ids by default.
        :param machine_probs:
        :param activate_ids:
        :return:
        """
        if activate_ids is None:
            self.machine_probs = machine_probs # （480，2）
        else:
            self.machine_probs = [.0] * self.data_len
            self.activate_data_ids = activate_ids
            prob_i = 0
            self.activate_data_idx = []
            for _activated_id in self.activate_data_ids:
                _activated_idx = self.id_2_index.get(_activated_id)
                self.machine_probs[_activated_idx] = machine_probs[prob_i]
                self.activate_data_idx.append(_activated_idx)
                prob_i += 1
        # self.machine_labels = utils.get_predict_label(self.machine_probs)
        if not (machine_labels is None):
            self.machine_labels = machine_labels    # （480）
        else:
            self.machine_labels = utils.get_predict_labels(self.data_ids)
        self.risk_labels = np.array([0] * self.data_len)
        self.risk_mul_labels = np.array([[1 for i in range(self.class_num)] for j in range(self.data_len)])
        self.machine_label_2_one = np.array([[0 for i in range(self.class_num)] for j in range(self.data_len)])
        self.risk_activate = np.array([[0 for i in range(self.class_num)] for j in range(self.data_len)])
        self.id_2_probs = dict()
        self.id_2_risk_labels = dict()

        for i in range(self.class_num):
            for j in range(self.data_len):
                if self.machine_labels[j] != i:
                    self.machine_label_2_one[j][i] = 0
                else:
                    self.machine_label_2_one[j][i] = 1

        for i in range(self.data_len):
            self.risk_activate[i][self.machine_labels[i]] = 1
            if self.machine_labels[i] != self.true_labels[i]:
                self.risk_labels[i] = 1
                # self.risk_activate[i][self.machine_labels[i]] = 1
                # self.risk_activate[i][self.true_labels[i]] = 1
            self.id_2_probs[self.data_ids[i]] = [self.machine_probs[i]]
            self.id_2_risk_labels[self.data_ids[i]] = self.risk_labels[i]
            if self.risk_labels[i] == 1:
                self.risk_mul_labels[i][self.true_labels[i]] = 0
            else:
                self.risk_mul_labels[i][self.true_labels[i]] = 0
            # if self.risk_labels[i] == 1:
            #     self.risk_mul_labels[i][self.true_labels[i]] = 1
            #     self.risk_mul_labels[i][self.machine_labels[i]] = 1

    def update_activate_matrix(self, effective_rule_idx, rules):

        rule_activate_matrix = []
        for i in range(self.data_len):
            rule_activate_matrix.append(np.array(self.original_rule_activation_matrix[i])[effective_rule_idx])
        rule_activation_matrix = np.array(rule_activate_matrix)

        class_idx = dict()
        idx = [0] * (self.class_num + 5)
        for i in range(len(rules)):
            _infer_class = rules[i].infer_class
            _infer_class = int(_infer_class[1:])
            idx[_infer_class + 1] = i + 1
        for i in range(1, self.class_num + 1):
            class_idx[i - 1] = [idx[i - 1], idx[i]]
            self.rules_max = max(self.rules_max, idx[i] - idx[i - 1])
        print('--each numbers of rules--')
        print(idx)
        print('--the max of class rules--')
        print(self.rules_max)

        new_rule_activate_matrix = []

        for i in range(self.data_len):
            _shape_2_4_3 = [[]]
            for j in range(len(rules)):
                rule_class = int(rules[j].infer_class[1:])

                if len(_shape_2_4_3) <= rule_class:
                    _shape_2_4_3.append([rule_activation_matrix[i][j]])

                    for k in range(len(_shape_2_4_3[rule_class - 1]), self.rules_max):
                        _shape_2_4_3[rule_class - 1].append(0)
                else:
                    _shape_2_4_3[rule_class].append(rule_activation_matrix[i][j])

            for j in range(len(_shape_2_4_3[self.class_num - 1]), self.rules_max):
                _shape_2_4_3[self.class_num - 1].append(0)

            """
            某次运行程序，发现_shape_2_4_3没有对齐，出错，因此加上下方的代码强制对齐
            """

            for i in range(len(_shape_2_4_3)):
                if len(_shape_2_4_3[i]) > self.rules_max:
                    _shape_2_4_3[i] = _shape_2_4_3[i][:self.rules_max]

            new_rule_activate_matrix.append(_shape_2_4_3)

        self.rule_activation_matrix = np.array(new_rule_activate_matrix)
        path = f'./results/CUB/{cfg.data_name}/{global_risk_dataset}'
        if not os.path.exists(path):
          os.makedirs(path)
        np.save(path + '/' + self.data_name + ' rule_activate.npy', self.rule_activation_matrix)

    def update_rule_matrix(self, id_2_pair_info, attr_2_index, rules):
        """

        :param id_2_pair_info: {id: [id, true label, attr1, attr2, attr3, ...]}
        :param attr_2_index: {attr1: 2, attr2: 3, attr3: 4}.
        :param rules: refer to class Rule in 'data_process/rules_process.py'.
        :return:
        """

        self.original_rule_activation_matrix = []
        # _count = 0
        # _interval = int(np.maximum(0.01 * self.data_len, 1))
        # _start_time = time.time()
        for i in tqdm(range(self.data_len), desc='Apply rules on {} data'.format(self.data_name)):
            v = id_2_pair_info.get(self.data_ids[i])

            rule_indicator = []
            for rule in rules:

                required_attrs_values = dict()
                for attr in rule.involved_attributes:
                    required_attrs_values[attr] = v[attr_2_index[attr]]
                temp = rule.apply(required_attrs_values)
                rule_indicator.append(temp)
            self.original_rule_activation_matrix.append(rule_indicator)

        self.original_rule_activation_matrix = np.array(self.original_rule_activation_matrix)

    def update_rule_features(self, mu_vector,
                             sigma_vector):
        # ---- Rule feature matrix. ----
        # rule_activation_matrix = []
        # for i in range(self.data_len):
        #    rule_activation_matrix.append(np.array(self.original_rule_activation_matrix[i])[effective_rule_index])
        # self.rule_activation_matrix = np.array(self.original_rule_activation_matrix)
        # risk_x = sp.lil_matrix(self.original_rule_activation_matrix)
        risk_x = self.rule_activation_matrix
        # -- Use scipy sparse matrix. --
        self.risk_mean_X_discrete = risk_x * np.array(mu_vector)
        self.risk_variance_X_discrete = risk_x * np.array(sigma_vector)

    def update_probability_feature(self, prob_interval_boundary_pts, ):

        # results = sbf.get_mul_probability_input_X(self.id_2_mul_probs, self.class_num,
        #                                           self.data_ids,
        #                                           prob_interval_boundary_pts,
        #                                           prob_dist_mean,
        #                                           prob_dist_variance)

        # 2020-07-24 just want to get mu and activate
        results = sbf.get_mul_probability_input_X(self.id_2_mul_probs,
                                                  self.class_num,
                                                  self.data_ids,
                                                  prob_interval_boundary_pts, )

        '''
        results = sbf.get_probability_input_X(self.id_2_probs, [0],
                                              self.data_ids,
                                              prob_interval_boundary_pts,
                                              prob_dist_mean,
                                              prob_dist_variance)
        '''
        self.risk_mean_X_continue = results[0]
        # self.risk_variance_X_continue = results[1]
        self.prob_activation_matrix = results[1]

    def get_mean_x(self):
        return sp.hstack((self.risk_mean_X_discrete, self.risk_mean_X_continue))

    def get_variance_x(self):
        return sp.hstack((self.risk_variance_X_discrete, self.risk_variance_X_continue))

    def get_activation_matrix(self):
        return sp.hstack((self.rule_activation_matrix, self.prob_activation_matrix))

    def get_risk_mean_X_discrete(self):
        return np.array(self.risk_mean_X_discrete)

    def get_risk_mean_X_continue(self):
        return np.array(self.risk_mean_X_continue)

    def get_risk_variance_X_discrete(self):
        return np.array(self.risk_variance_X_discrete)

    def get_risk_variance_X_continue(self):
        return np.array(self.risk_variance_X_continue)

    def get_rule_activation_matrix(self):
        return np.array(self.rule_activation_matrix)

    def get_prob_activation_matrix(self):
        return np.array(self.prob_activation_matrix)

    def print_evaluation_info(self):
        if self.machine_labels is None:
            logging.info("Warning: No machine results are provided!")
        else:
            _precision = precision_score(self.true_labels, self.machine_labels)
            _recall = recall_score(self.true_labels, self.machine_labels)
            _f1 = f1_score(self.true_labels, self.machine_labels)
            print("- {} data's Precision: {}, Recall: {}, F1-Score: {}.".format(self.data_name,
                                                                                _precision,
                                                                                _recall,
                                                                                _f1))

    def print_rules_coverage(self):
        print("Coverage BEFORE filtering rules whose train data size is small.")
        print(self.original_rule_activation_matrix)
        data_no_rules_num = np.sum(self.original_rule_activation_matrix, axis=1).tolist().count(0)
        rule_cover_num = self.data_len - data_no_rules_num
        print("- # {} data has rules / # {} data = {} / {}, coverage = {:.2%}".format(self.data_name, self.data_name,
                                                                                      rule_cover_num, self.data_len,
                                                                                      1.0 * rule_cover_num / self.data_len))
        print("Coverage AFTER filtering rules whose train data size is small.")
        data_no_rules_num = np.sum(self.rule_activation_matrix, axis=1).tolist().count(0)
        rule_cover_num = self.data_len - data_no_rules_num
        print("- # {} data has rules / # {} data = {} / {}, coverage = {:.2%}".format(self.data_name, self.data_name,
                                                                                      rule_cover_num, self.data_len,
                                                                                      1.0 * rule_cover_num / self.data_len))


def load_data(cfg, data_name, risk_dataset_rate):
    """
    :param cfg: the Configuration class. refer to 'Common/config.py'.
    :return:
    """

    print("- Data source: {}.".format(cfg.get_all_data_path())) # all_data_info.csv
    df = pd.read_csv(cfg.get_all_data_path())
    pairs = df.values
    print("- # of data: {}.".format(len(pairs)))
    id_2_pair_info = dict()
    for elem in pairs:
        id_2_pair_info[elem[0]] = elem  # <id, info>

    # -- Train data --
    train_info = pd.read_csv(os.path.join(cfg.get_parent_path(), 'train.csv')).values
    train_ids = train_info[:, 0].astype(str)
    train_data = DataInfo(train_ids, id_2_pair_info, name='Training')
    print("- # of training data: {}.".format(len(train_ids)))

    # fine-tune data
    # -- Validation data --
    #valida_info = pd.read_csv(os.path.join(cfg.get_parent_path(), 'val.csv')).values
    valida_info = pd.read_csv(os.path.join(cfg.get_parent_path(), 'val.csv')).values
    valida_ids = valida_info[:, 0].astype(str)
    validation_data = DataInfo(valida_ids, id_2_pair_info, name='Validation')
    print("- # of validation data: {}.".format(len(valida_ids)))

    # -- Test data --
    # test_info = pd.read_csv(os.path.join(cfg.get_parent_path(), 'test.csv')).values
    test_info = pd.read_csv(os.path.join(cfg.get_parent_path(), 'test.csv')).values
    test_ids = test_info[:, 0].astype(str)
    test_data = DataInfo(test_ids, id_2_pair_info, name='Test')
    print("- # of test data: {}.".format(len(test_ids)))

    # -- Loading rules --
    attr_2_index = dict()
    i = 2
    while i < len(df.columns):
        attr_2_index[df.columns[i]] = i
        i += 1

    rules = rp.read_rules(cfg.get_decision_tree_rules_path()) # decision_tree_rules_clean.txt

    # rules = rp.clean_rules_mt(rules, 5)

    rule_class = []
    for i in range(len(rules)):
        rule_class.append(rules[i].infer_class)

    # -- Apply rules on data --
    train_data.update_rule_matrix(id_2_pair_info, attr_2_index, rules)
    # np.save('train.npy', train_data.original_rule_activation_matrix)

    validation_data.update_rule_matrix(id_2_pair_info, attr_2_index, rules)
    # np.save('val.npy', validation_data.original_rule_activation_matrix)

    test_data.update_rule_matrix(id_2_pair_info, attr_2_index, rules)
    # np.save('test.npy', test_data.original_rule_activation_matrix)

    # -- Calculate the distribution of rule features --
    # 1. the numbers < minimum_observation_num
    # 2. the acc < rule_acc
    # 3. the rule similar, which is handled when generates the rules
    new_rules = []
    new_rules_class = []
    _minimum_observations = cfg.minimum_observation_num
    data_id_2_rules = np.array(train_data.original_rule_activation_matrix)
    val_data_id_2_rules = np.array(validation_data.original_rule_activation_matrix)
    test_data_id_2_rules = np.array(test_data.original_rule_activation_matrix)
    rule_2_data_ids = data_id_2_rules.T
    vla_rule_2_data_ids = val_data_id_2_rules.T
    test_rule_2_data_ids = test_data_id_2_rules.T
    effective_rule_index = []

    for rule_i in range(len(rules)):
        data_index_4_rule_i = np.where(rule_2_data_ids[:][rule_i] == 1)[0]
        if rules[rule_i].infer_class[0] == 'M':
            if len(data_index_4_rule_i) < train_data.data_len * (1. / cfg.get_class_num()) * _minimum_observations:
                continue
        else:
            if len(data_index_4_rule_i) < train_data.data_len * ((cfg.get_class_num() - 1) /
                                                                 cfg.get_class_num()) * _minimum_observations:
                continue

        right = 0.
        tot = 0.
        for val_i in range(validation_data.data_len):
            if vla_rule_2_data_ids[rule_i][val_i] == 1:
                if rule_class[rule_i][0] == 'M':
                    if validation_data.true_labels[val_i] == int(rule_class[rule_i][1:]):
                        right += 1
                else:
                    if validation_data.true_labels[val_i] != int(rule_class[rule_i][1:]):
                        right += 1
                tot += 1

        if tot > 0:
            # print('the rule {} acc: {}/{}={}'.format(rule_i, right, tot, right/tot))
            path = os.path.join('./{}/{}'.format(data_name, risk_dataset_rate), 'rule_acc.txt')
            with open(path, 'a') as f:
                f.write('the val rule {} acc: {}/{}={}\n'.format(rule_i, right, tot, right / tot))
        if tot < 1:
            continue
        if right / tot < cfg.rule_acc:
            logging.info(rules[rule_i].infer_class)
            logging.info(right / tot)
            continue

        right = 0
        tot = 0

        for test_i in range(test_data.data_len):
            if test_rule_2_data_ids[rule_i][test_i] == 1:
                if rule_class[rule_i][0] == 'M':
                    if test_data.true_labels[test_i] == int(rule_class[rule_i][1:]):
                        right += 1
                else:
                    if test_data.true_labels[test_i] != int(rule_class[rule_i][1:]):
                        right += 1
                tot += 1

        if tot > 0:
            # print('the rule {} acc: {}/{}={}'.format(rule_i, right, tot, right/tot))
            with open(path, 'a') as f:
                f.write('the test rule {} acc: {}/{}={}\n'.format(rule_i, right, tot, right / tot))

        effective_rule_index.append(rule_i)
        new_rules.append(rules[rule_i])
        new_rules_class.append(rule_class[rule_i])

    # 写出筛选后保留的规�?
    path = os.path.join('./{}/{}'.format(data_name, risk_dataset_rate), 'new_rules.txt')
    with open(path, 'w') as f:  # 根目录

        for rule in new_rules:
            f.write(rule.readable_description)
            f.write('\n')

    train_data.update_activate_matrix(effective_rule_index, new_rules)
    validation_data.update_activate_matrix(effective_rule_index, new_rules)
    test_data.update_activate_matrix(effective_rule_index, new_rules)
    # np.save('train_rules_activate.npy', train_data.get_rule_activation_matrix())
    # np.save('val_rules_activate.npy', validation_data.get_rule_activation_matrix())
    # np.save('test_rules_activate.npy', test_data.get_rule_activation_matrix())

    rules_distribution = []

    # 计算规则的均值及方差�?此处�?分类相比改动较大�?目的一�?
    for i in range(cfg.get_class_num()):
        rules_distribution.append([])
        for j in range(train_data.rules_max):

            temp = train_data.rule_activation_matrix[:, i, j]

            mu_sigma = utils.calculate_rules_feature_mu_sigma(np.array(train_data.data_ids),
                                                              train_data.rule_activation_matrix[:, i, j], i,
                                                              train_data.id_2_true_labels)
            if math.isnan(mu_sigma[0]) or math.isnan(mu_sigma[1]):
                mu_sigma = [0, 0]
            # mu_sigma = [0,0]
            rules_distribution[i].append(mu_sigma)
    rules_distribution = np.array(rules_distribution)
    mu_vector = rules_distribution[:, :, 0].reshape([cfg.get_class_num(), train_data.rules_max])
    sigma_vector = rules_distribution[:, :, 1].reshape([cfg.get_class_num(), train_data.rules_max])
    train_data.mu_vector = mu_vector
    train_data.sigma_vector = sigma_vector

    # -- update rule features of data --
    train_data.update_rule_features(mu_vector, sigma_vector)
    validation_data.update_rule_features(mu_vector, sigma_vector)
    test_data.update_rule_features(mu_vector, sigma_vector)

    # 此处�?分类覆盖率计算方法，对多分类，存在问�?
    # train_data.print_rules_coverage()
    # validation_data.print_rules_coverage()
    # test_data.print_rules_coverage()

    return train_data, validation_data, test_data
