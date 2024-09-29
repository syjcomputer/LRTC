from torch.distributions import Normal
from torch.nn.functional import softmax
from torch.nn.modules.loss import _Loss

from RiskAnalysis.data_process import similarity_based_feature as sbf
from RiskAnalysis.common import config
import numpy as np
from . import torch_learn_weights as torchlearn
from collections import Counter
import torch
import torch.distributed as dist
# import torch
# from torch.nn.modules.loss import _Loss
# from torch.distributions.normal import Normal
import os
cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)
class_num = cfg.get_class_num()
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
device = [0, 1] # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RiskTorchModel(object):
    def __init__(self):
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
        self.learn_confidence = 0.9
        self.match_value = None
        self.unmatch_value = None
        # train data
        self.train_data = None
        # validation data
        self.validation_data = None
        # test data
        self.test_data = None
        self.model = None

    #def train(self, train_machine_probs, valida_machine_probs, train_machine_mul_probs, valida_machine_mul_probs,):
    def train(self, train_machine_probs, valida_machine_probs, test_machine_probs, train_machine_mul_probs, valida_machine_mul_probs, test_machine_mul_probs,
              train_labels=None, val_labels=None, test_labels=None):
        # use new classifier output probabilities.

        self.train_data.update_machine_info(train_machine_probs, train_labels)
        self.validation_data.update_machine_info(valida_machine_probs, val_labels)
        self.test_data.update_machine_info(test_machine_probs, test_labels)
        self.train_data.update_machine_mul_info(train_machine_mul_probs)
        self.validation_data.update_machine_mul_info(valida_machine_mul_probs)
        self.test_data.update_machine_mul_info(test_machine_mul_probs)

        # update the probability feature of training data
        self.train_data.update_probability_feature(self.prob_interval_boundary_pts,)
        # update the probability feature of validation data
        self.validation_data.update_probability_feature(self.prob_interval_boundary_pts,)
        self.test_data.update_probability_feature(self.prob_interval_boundary_pts)

        _feature_activation_matrix = np.concatenate(
            (np.array(self.validation_data.get_rule_activation_matrix()), np.array(self.validation_data.get_prob_activation_matrix())), axis=2)

        max_w = 1.0 / np.max(np.sum(_feature_activation_matrix, axis=2))
        class_count = Counter(self.validation_data.machine_labels.reshape(-1))
        # dist.init_process_group(backend='gloo', init_method='env://')
        model = torchlearn.RiskModel(self.validation_data.get_risk_mean_X_discrete().shape[2], max_w).cuda()
        # model = torch.nn.DataParallel(model)
        self.model = model.cuda()
        # for i in self.model.parameters():
        #     print(i)
        func_params, rule_w, rule_var, machine_var = torchlearn.train(self.model, self.validation_data, self.test_data)
        self.func_params = func_params
        self.rule_w = rule_w
        self.rule_var = rule_var
        self.machine_var = machine_var




    def predict(self, test_machine_probs, test_machine_mul_probs):
        # use new classifier output probabilities.

        # self.test_data.update_machine_info(test_machine_probs)
        # self.test_data.update_machine_mul_info(test_machine_mul_probs)
        # # update the probability feature of training data
        # self.test_data.update_probability_feature(self.prob_interval_boundary_pts,
        #                                           self.prob_dist_mean,
        #                                           self.prob_dist_variance)
        # self.test_data = self.validation_data
        results = torchlearn.predict(self.model, self.test_data, True)
        predict_probs = results
        self.test_data.risk_values = predict_probs


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



class RiskLoss(_Loss):
    def __init__(self, risk_model, size_average=None, reduce=None, reduction='mean'):
        super(RiskLoss, self).__init__(size_average, reduce, reduction)
        self.LEARN_VARIANCE = cfg.learn_variance
        # self.rm = risk_model
        self.a = torch.tensor(0., dtype=torch.float).to(device[0])
        self.b = torch.tensor(1., dtype=torch.float).to(device[0])
        self.alpha = torch.tensor(risk_model.learn_confidence, dtype=torch.float).to(device[0])
        self.weight_func_a = risk_model.func_params[0].to(device[0])
        self.weight_func_b = risk_model.func_params[1].to(device[0])
        self.weight_func_c = risk_model.func_params[2].to(device[0])
        self.m = -1
        self.continuous_m = cfg.interval_number_4_continuous_value
        self.discrete_m = -1
        self.rule_w = risk_model.rule_w.to(device[0])
        self.rule_var = risk_model.rule_var.to(device[0])
        self.machine_var = risk_model.machine_var.to(device[0])
        self.reduction = 'mean'
        # print(self.weight_func_a)
        # print(self.weight_func_b)
        # print(self.weight_func_c)


    def forward(self, machine_lables, rule_mus, machine_mus,
                rule_feature_matrix, machine_feature_matrix, machine_one, outputs):

        machine_pro = softmax(outputs.to(device[0]), dim=1)

        # rule_w = self.rule_w.clamp(0, 1)
        # rule_var = self.rule_var.clamp(1e-10, 1)
        # machine_var = self.machine_var.clamp(1e-10, 1)

        machine_mus_vector = torch.reshape(torch.sum(machine_mus, 2), (-1, class_num)).to(device[0])
        machine_w = gaussian_function(self.weight_func_a, self.weight_func_b, self.weight_func_c, machine_mus_vector)


        big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * machine_w + 1e-10

        rule_sigma = rule_feature_matrix * self.rule_var

        machine_sigma = machine_feature_matrix * self.machine_var
        machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, class_num))

        big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (machine_w ** 2) + 1e-10

        weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) + machine_w + 1e-10

        big_mu = big_mu / weight_vector
        big_sigma = big_sigma / (weight_vector ** 2)



        Fr_alpha = my_truncated_normal_ppf(self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alpha, self.a, self.b, big_mu, torch.sqrt(big_sigma))


        # prob = - (1. - Fr_alpha) * torch.log(torch.ones_like(Fr_alpha).to(device[0]) - machine_pro) - (1. - (
        #         torch.ones_like(Fr_alpha_bar).to(device[0]) - Fr_alpha_bar)) * torch.log(machine_pro)

        # prob = - (1. - (torch.ones_like(Fr_alpha_bar).to(device[0]) - Fr_alpha_bar)) * torch.log(machine_pro)

        prob = torch.sum(- ((1. -  Fr_alpha)) * torch.log(torch.ones_like(machine_pro).to(device[0]) - machine_pro + 1e-10) - (1. - (
            torch.ones_like(Fr_alpha_bar).to(device[0]) - Fr_alpha_bar)) * torch.log(machine_pro + 1e-10), 1) / class_num
        # prob = -torch.sum(Fr_alpha_bar * torch.log(machine_pro + 1e-10), 1) / class_num
        # print('{} {} {} {}'.format(1 - Fr_alpha, torch.log(torch.ones_like(machine_pro).to(device[0]) - machine_pro),
        #                            Fr_alpha_bar, torch.log(machine_pro)))
        # print(prob)

        # prob = torch.sum(- (1. - Fr_alpha) * torch.log(torch.ones_like(machine_pro).to(device[0]) - machine_pro + 1e-10), 1) / class_num
        #
        # print('--------------------------FR')
        # print(Fr_alpha)
        # print(1 - Fr_alpha_bar)
        # prob = torch.sum(
        #     (- (1. - Fr_alpha) * torch.log(torch.ones_like(machine_pro).to(device[0]) - machine_pro) - (1. - (
        #             torch.ones_like(Fr_alpha_bar).to(device[0]) - Fr_alpha_bar)) * torch.log(machine_pro)) * y_activate,
        #     1)

        # A Sigmoid mapping for the output probability.
        # machine_label = 1.0 / (1.0 + torch.exp(- 100 * (machine_label - 0.5)))
        # prob = Fr_alpha * (torch.ones_like(Fr_alpha).to(device[0]) - machine_label) + (
        #         torch.ones_like(Fr_alpha_bar).to(device[0]) - Fr_alpha_bar) * machine_label
        # print("big mu:")
        # print(big_mu)
        # print("mu labels:")
        # temp_labels = []
        # for elem in big_mu:
        #     if elem >= 0.5:
        #         temp_labels.append(1)
        #     else:
        #         temp_labels.append(0)
        # temp_labels = np.array(temp_labels)
        # print(temp_labels)
        # print("VaR-:")
        # print(Fr_alpha)
        # print("VaR+:")
        # print(1.0 - Fr_alpha_bar)
        # print("risk values:")
        # print(prob)

        # Weighted loss according to the indicated label by the expectation
        # print("Set instance weights:")
        # weights = Counter(temp_labels)
        # t_instance_weights = np.array([1.0] * len(temp_labels))
        # w_pos = weights.get(1)
        # if w_pos is None:
        #     w_pos = 0
        # w_neg = weights.get(0)
        # if w_neg is None:
        #     w_neg = 0
        # w_normalization = w_pos + w_neg + 2
        # t_instance_weights[np.where(temp_labels == 0)] = (w_pos + 1) / w_normalization
        # t_instance_weights[np.where(temp_labels == 1)] = (w_neg + 1) / w_normalization
        # print(t_instance_weights)
        # instance_weight = torch.from_numpy(t_instance_weights)
        instance_weight = None
        if instance_weight is not None:
            instance_weight = instance_weight.to(device[0])
            prob = prob * instance_weight

        if self.reduction != 'none':
            if instance_weight is not None:
                ret = torch.sum(prob) / torch.sum(instance_weight) if self.reduction == 'mean' else torch.sum(prob)
            else:
                ret = torch.mean(prob) if self.reduction == 'mean' else torch.sum(prob)

        return ret.type(torch.FloatTensor).to(device[0])
