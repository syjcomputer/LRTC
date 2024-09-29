from torch.distributions import Normal
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss

from RiskAnalysis.data_process import similarity_based_feature as sbf
from RiskAnalysis.common.config import cfg, global_data
import numpy as np
from . import torch_learn_weights as torchlearn
from collections import Counter
import torch
import torch.distributed as dist
# import torch
# from torch.nn.modules.loss import _Loss
# from torch.distributions.normal import Normal
import os

# cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection,
#                            config.global_risk_dataset)
class_num = cfg.get_class_num()

# device = [0,1]  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device2 = [0,1]

device = [torch.device('cuda:0'), torch.device('cpu')]
device2 = [torch.device('cuda'), torch.device('cpu')]


class RiskTorchModel(object):
    def __init__(self, learn_confidence=0.9, learning_rate=0.01, bs=16):
        self.prob_interval_boundary_pts = sbf.get_equal_intervals(0.0, 1.0, cfg.interval_number_4_continuous_value)
        self.prob_dist_mean = None
        self.prob_dist_variance = None
        # parameters for risk model
        self.learn_weights = None
        self.rule_w = None
        self.func_params = [0.5, 0.5, 1]
        self.rule_var = None
        self.machine_var = None
        self.a = 0.0
        self.b = 1.0
        self.learn_confidence = learn_confidence
        self.match_value = None
        self.unmatch_value = None
        # train data
        self.train_data = None
        # validation data
        self.validation_data = None
        # test data
        self.test_data = None
        self.model = None
        self.init_rule_mu = None
        self.init_machine_mu = None

        # torch_model.train
        self.learning_rate = learning_rate
        self.bs = bs
        # self.device = device

    # def train(self, train_machine_probs, valida_machine_probs, train_machine_mul_probs, valida_machine_mul_probs,):
    def train(self, train_machine_probs, valida_machine_probs, test_machine_probs, train_machine_mul_probs,
              valida_machine_mul_probs, test_machine_mul_probs,
              train_labels=None, val_labels=None, test_labels=None, epoch="", epoches=5, store_name="20News", test_loss=0):
        # use new classifier output probabilities.


        self.train_data.update_machine_info(train_machine_probs, train_labels)
        self.validation_data.update_machine_info(valida_machine_probs, val_labels)
        self.test_data.update_machine_info(test_machine_probs, test_labels)

        self.train_data.update_machine_mul_info(train_machine_mul_probs)
        self.validation_data.update_machine_mul_info(valida_machine_mul_probs)
        self.test_data.update_machine_mul_info(test_machine_mul_probs)
        '''
        prob_interval_2_ids, class_label = sbf.get_mul_continuous_interval_to_ids(self.train_data.id_2_mul_probs,
                                                                                  class_num,
                                                                                  self.train_data.data_ids,
                                                                                  self.prob_interval_boundary_pts)

        self.prob_dist_mean, self.prob_dist_variance = sbf.calculate_mul_similarity_interval_distributions(
            prob_interval_2_ids,
            class_label,
            self.train_data.id_2_true_labels,
            cfg.minimum_observation_num)
        '''
        self.prob_dist_mean = ([1.] * cfg.interval_number_4_continuous_value) * (
            np.linspace(0, 1, cfg.interval_number_4_continuous_value + 1)[1:]) / 2

        # print(self.prob_dist_variance.shape, self.prob_dist_mean.shape)
        # print(self.prob_dist_variance)
        # print('---------mean')
        # print(self.prob_dist_mean)

        init_rule_mu = self.train_data.mu_vector
        init_machine_mu = self.prob_dist_mean
        init_rule_mu[init_rule_mu < 0] = 0
        init_machine_mu[init_machine_mu < 0] = 0
        self.init_rule_mu = init_rule_mu
        self.init_machine_mu = init_machine_mu
        # print('------update')
        # print(init_rule_mu)
        # print(init_machine_mu)

        # update the probability feature of training data
        self.train_data.update_probability_feature(self.prob_interval_boundary_pts, )
        # update the probability feature of validation data
        self.validation_data.update_probability_feature(self.prob_interval_boundary_pts, )
        self.test_data.update_probability_feature(self.prob_interval_boundary_pts)

        _feature_activation_matrix = np.concatenate(
            (np.array(self.validation_data.get_rule_activation_matrix()),
             np.array(self.validation_data.get_prob_activation_matrix())), axis=2)

        max_w = 1.0 / np.max(np.sum(_feature_activation_matrix, axis=2))
        class_count = Counter(self.validation_data.machine_labels.reshape(-1))
        # dist.init_process_group(backend='gloo', init_method='env://')
        print('------------------------------- load RiskModel')
        model = torchlearn.RiskModel(self.validation_data.get_risk_mean_X_discrete().shape[2], max_w).to(
            device[0])  # (device[0], self.learn_confidence)
        # model = torch.nn.DataParallel(model)
        self.model = model
        # for i in self.model.parameters():
        #     print(i)
        func_params, rule_w, rule_var, machine_var = torchlearn.train(self.model, self.validation_data, self.test_data,
                                                                      init_rule_mu, init_machine_mu, epoch, epoches,
                                                                      self.learning_rate, self.bs,True, store_name, test_loss)
        self.func_params = func_params
        self.rule_w = rule_w
        self.rule_var = rule_var
        self.machine_var = machine_var

    def predict(self, test_machine_probs, test_machine_mul_probs, epoch_cnn):
        # use new classifier output probabilities.

        # self.test_data.update_machine_info(test_machine_probs)
        # self.test_data.update_machine_mul_info(test_machine_mul_probs)
        # # update the probability feature of training data
        # self.test_data.update_probability_feature(self.prob_interval_boundary_pts,
        #                                           self.prob_dist_mean,
        #                                           self.prob_dist_variance)
        # self.test_data = self.validation_data
        results, risk_roc_auc = torchlearn.predict(global_data, self.model, self.test_data, 0, self.init_rule_mu,
                                                   self.init_machine_mu, epoch_cnn, True)
        predict_probs = results
        self.test_data.risk_values = predict_probs

        return risk_roc_auc


def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    x = torch.zeros_like(mean)
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    norm = Normal(mean, stddev)
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    y = torch.where(a > 0,
                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
    return torch.reshape(y, (-1, class_num))


def gaussian_function(a, b, c, x):
    _part = (- torch.div((x - b) ** 2, 2.0 * (c ** 2)))
    _f = -torch.exp(_part) + a + 1.0
    return _f


def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    x = torch.zeros_like(mean)
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    # norm是class*data_nums个正态分布
    norm = Normal(mean, stddev)
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    # 被标记为某一类别的概率
    y = torch.where(a > 0,
                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
    return torch.reshape(y, (-1, class_num))


def gaussian_function(a, b, c, x):
    _part = (- torch.div((x - b) ** 2, 2.0 * (c ** 2)))
    _f = -torch.exp(_part) + a + 1.0
    return _f


class RiskLoss(_Loss):
    def __init__(self, risk_model, size_average=None, reduce=None, reduction='mean'):
        super(RiskLoss, self).__init__(size_average, reduce, reduction)
        self.LEARN_VARIANCE = cfg.learn_variance
        # self.rm = risk_model
        self.a = torch.tensor(0., dtype=torch.float64).to(device2[0])
        self.b = torch.tensor(1., dtype=torch.float64).to(device2[0])
        self.alpha = torch.tensor(risk_model.learn_confidence, dtype=torch.float64).to(device2[0])
        self.weight_func_a = risk_model.func_params[0].to(device2[0])
        self.weight_func_b = risk_model.func_params[1].to(device2[0])
        self.weight_func_c = risk_model.func_params[2].to(device2[0])
        self.m = -1
        self.continuous_m = cfg.interval_number_4_continuous_value
        self.discrete_m = -1
        self.rule_w = risk_model.rule_w.to(device2[0])
        self.rule_var = risk_model.rule_var.to(device2[0])
        self.machine_var = risk_model.machine_var.to(device2[0])
        self.reduction = 'mean'
        del risk_model

    def forward(self, machine_lables, rule_mus, machine_mus,
                rule_feature_matrix, machine_feature_matrix, machine_one, outputs, labels=None):
        machine_pro = softmax(outputs.to(device2[0]), dim=1)
        machine_pro = machine_pro.to(dtype=torch.float64)
        # print(machine_pro.shape)
        # rule_w = self.rule_w.clamp(0, 1)
        # rule_var = self.rule_var.clamp(1e-10, 1)
        # machine_var = self.machine_var.clamp(1e-10, 1)

        machine_mus_vector = torch.reshape(torch.sum(machine_mus, 2), (-1, class_num)).to(device2[0])
        # machine_mus_vector = torch.reshape(machine_pro, (-1, class_num)).to(device2[0])
        # machine_w = gaussian_function(self.weight_func_a.to(torch.float64), self.weight_func_b.to(torch.float64), self.weight_func_c.to(torch.float64), machine_mus_vector.reshape(-1, class_num))
        machine_w = gaussian_function(self.weight_func_a.to(torch.float64), self.weight_func_b.to(torch.float64),
                                      self.weight_func_c.to(torch.float64), machine_mus_vector.reshape(-1, class_num))
        machine_w = machine_w.reshape((-1, class_num))

        big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * machine_w + 1e-10
        # big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * torch.sum(rule_feature_matrix * self.rule_w, 2) + 1e-10

        # new_machine_feature_matrix = torch.zeros_like(machine_feature_matrix, requires_grad=False)
        # for i in range(len(new_machine_feature_matrix)):
        #     for j in range(len(new_machine_feature_matrix[0])):
        #         x = int(99) if int(machine_pro[i][j] * 100) == 100 else int(machine_pro[i][j] * 100)
        #         new_machine_feature_matrix[i][j][x] = 1
        # machine_sigma = new_machine_feature_matrix * self.machine_var
        # machine_sigma = machine_feature_matrix * self.machine_var
        # machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, class_num))

        # RSD
        # new_rule_mus = rule_mus.clone()
        # new_mac_mus = machine_mus.clone()
        # new_rule_mus[torch.where(new_rule_mus < 1e-10)] = 0.1
        # new_mac_mus[torch.where(new_mac_mus < 1e-10)] = 0.1
        # rule_standard_deviation = new_rule_mus * self.rule_var
        # mac_standard_deviation = new_mac_mus * self.machine_var
        # rule_var = rule_standard_deviation ** 2
        # machine_var = mac_standard_deviation ** 2
        # print(self.rule_var.shape)
        rule_sigma = rule_feature_matrix * self.rule_var

        machine_sigma = machine_feature_matrix * self.machine_var
        machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, class_num))

        big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (machine_w ** 2) + 1e-10
        # big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (torch.sum(rule_feature_matrix * self.rule_w, 2) ** 2) + 1e-10
        weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) + machine_w + 1e-10

        big_mu = big_mu / weight_vector
        big_sigma = big_sigma / (weight_vector ** 2)

        Fr_alpha = my_truncated_normal_ppf(self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))

        risk_label = torch.argmax(Fr_alpha_bar, 1)
        risk_label.requires_grad = False
        return risk_label
        '''
        risk_one_label = torch.zeros_like(machine_pro, requires_grad=False)
        for i in range(len(risk_one_label)):
            risk_one_label[i][risk_label[i]] = 1

        prob = torch.sum(- risk_one_label * torch.log(machine_pro + 1e-10), 1).reshape(-1, 1)
        # print(1 - Fr_alpha_bar)
        # pred = softmax(Fr_alpha_bar * 100, dim=1)
        # prob = torch.sum(- pred * torch.log(machine_pro + 1e-10), 1)
        # print('-----')
        # print(prob)
        # print('----------mac')
        # print(torch.argmax(machine_pro, 1))
        # print('----label')
        # print(labels)
        # print('max fr')
        # print(torch.argmax(Fr_alpha, 1))
        # print('min Fr_bar')
        # print(torch.argmax(pred, 1))
        # print(torch.max(pred, 1))

        instance_weight = None
        if instance_weight is not None:
            instance_weight = instance_weight.to(device2[0])
            prob = prob * instance_weight

        if self.reduction != 'none':
            if instance_weight is not None:
                ret = torch.sum(prob) / torch.sum(instance_weight) if self.reduction == 'mean' else torch.sum(prob)
            else:
                ret = torch.mean(prob) if self.reduction == 'mean' else torch.sum(prob)

        return ret.type(torch.FloatTensor).to(device2[0]), risk_label
        '''
