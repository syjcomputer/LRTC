import numpy as np
import math
from RiskAnalysis.common import utils
from scipy import sparse as sp


def get_equal_intervals(minimum_value=0.0, maximum_value=1.0, interval_number=10):
    return np.linspace(minimum_value, maximum_value, interval_number + 1)


def get_interval_index(real_number, interval_boundary_points):
    for i in range(len(interval_boundary_points) - 1):
        if interval_boundary_points[i] <= real_number < interval_boundary_points[i + 1]:
            return i
    # if the input equals to the maximum value, it is in the last interval.
    return len(interval_boundary_points) - 2


def calculate_mul_similarity_interval_distributions(sim_intervals_2_ids, class_label, id_2_labels,
                                                    _minimum_observations):
    mu_matrix = []
    sigma_matrix = []
    for _sim_index in range(len(sim_intervals_2_ids)):
        # print('------------------------------sim_intervals_2_ids')
        # print(len(sim_intervals_2_ids[0]))
        mu_matrix.append([])
        sigma_matrix.append([])
        v = sim_intervals_2_ids[_sim_index]
        # print('----------------------------------v')
        # print(len(v))
        # print(v)
        label = class_label[_sim_index]
        # print('--------label')
        # print(label)
        interval_mus = []
        interval_sigmas = []
        for i in range(len(v)):
            if len(v[i]) < _minimum_observations:
                interval_mus.append(-1)
                interval_sigmas.append(-1)
            else:
                mu_sigma = utils.calculate_machine_feature_mu_sigma(v[i], id_2_labels, label)
                if math.isnan(mu_sigma[0]) or math.isnan(mu_sigma[1]):
                    interval_mus.append(-1)
                    interval_sigmas.append(-1)
                else:
                    interval_mus.append(mu_sigma[0])
                    interval_sigmas.append(mu_sigma[1])

        mu_matrix[_sim_index].extend(interval_mus)
        sigma_matrix[_sim_index].extend(interval_sigmas)
    mu_matrix = np.array(mu_matrix)
    sigma_matrix = np.array(sigma_matrix)
    # print('---------------------------calculate_mul_similarity_interval_distributions')
    # print(mu_matrix)
    # print('sigma')
    # print(sigma_matrix)
    return mu_matrix, sigma_matrix


def calculate_similarity_interval_distributions(sim_intervals_2_ids, id_2_labels, id_2_risk_labels,
                                                _minimum_observations):
    """

    :param sim_intervals_2_ids: list() {sim_name1: [{ids}, {ids}, ...],
                                        sim_name2: [{ids}, {ids}, ...],
                                        ...}
    :param id_2_labels:
    :param _minimum_observations:
    :return: mean matrix: m features * n intervals; variance matrix: m features * n intervals.
    """
    mu_matrix = []
    sigma_matrix = []
    for _sim_index in range(len(sim_intervals_2_ids)):
        v = sim_intervals_2_ids[_sim_index]
        interval_mus = []
        interval_sigmas = []
        for i in range(len(v)):
            if len(v[i]) < _minimum_observations:
                interval_mus.append(-1)
                interval_sigmas.append(-1)
            else:
                mu_sigma = utils.calculate_feature_mu_sigma(v[i], id_2_labels, 'C', id_2_risk_labels)
                if mu_sigma is None:
                    interval_mus.append(-1)
                    interval_sigmas.append(-1)
                else:
                    if math.isnan(mu_sigma[0]) or math.isnan(mu_sigma[1]):
                        mu_sigma = [-1, -1]
                    interval_mus.append(mu_sigma[0])
                    interval_sigmas.append(mu_sigma[1])
                    # interval_mus.append(1)
                    # interval_sigmas.append(0)

        # print('----------------------------interval_mus')
        # print(interval_mus)
        mu_matrix.append(interval_mus)
        sigma_matrix.append(interval_sigmas)
    mu_matrix = np.array(mu_matrix)
    sigma_matrix = np.array(sigma_matrix)
    return mu_matrix, sigma_matrix


def get_mul_machine_mu_sigma(id2contvalue, class_num, train_ids, interval_boundary_points):
    sim_intervals_2_ids = []
    class_label = []

    interval_list = []
    for j in range(len(interval_boundary_points) - 1):
        interval_list.append(set())
    sim_intervals_2_ids.append(interval_list)

    for _id in train_ids:
        _pair_info = id2contvalue.get(_id)
        for i in range(class_num):
            _feature_value = _pair_info[0][i]
            _interval_index = get_interval_index(_feature_value, interval_boundary_points)
            sim_intervals_2_ids[i][_interval_index].add(_id)
            # print(_interval_index)
    return sim_intervals_2_ids, class_label


def get_mul_continuous_interval_to_ids(id2contvalue, class_num, train_ids, interval_boundary_points):
    sim_intervals_2_ids = []
    class_label = []
    for i in range(class_num):
        class_label.append('C' + str(i))
        interval_list = []
        for j in range(len(interval_boundary_points) - 1):
            interval_list.append(set())
        sim_intervals_2_ids.append(interval_list)

    for _id in train_ids:
        _pair_info = id2contvalue.get(_id)
        # print('----------------------_pair_info')
        # print(len(_pair_info))
        # print(_pair_info)
        for i in range(class_num):
            _feature_value = _pair_info[0][i]
            # print(_feature_value)
            _interval_index = get_interval_index(_feature_value, interval_boundary_points)
            sim_intervals_2_ids[i][_interval_index].add(_id)
            # print(_interval_index)
    return sim_intervals_2_ids, class_label


def get_continuous_interval_to_ids(id2contvalue, feature_index, train_ids, interval_boundary_points):
    """

    :param id2contvalue: dict(): {id: [], its continuous values, id: [], id: [], ...}
    :param feature_index: the index of selected features being applied.
    :param train_ids:
    :param interval_boundary_points:
    :return:
    """
    sim_intervals_2_ids = []
    for i in range(len(feature_index)):
        interval_list = []
        for j in range(len(interval_boundary_points) - 1):
            interval_list.append(set())
        sim_intervals_2_ids.append(interval_list)
    for _id in train_ids:
        _pair_info = id2contvalue.get(_id)
        for i in range(len(feature_index)):
            _f_index = feature_index[i]
            _feature_value = _pair_info[_f_index]
            _interval_index = get_interval_index(_feature_value, interval_boundary_points)
            sim_intervals_2_ids[i][_interval_index].add(_id)
    return sim_intervals_2_ids


def get_continuous_input_X(id2contvalue,
                           feature_index,
                           pair_ids,
                           interval_boundary_points,
                           mean_matrix,
                           sigma_matrix):
    """

    :param id2contvalue: dict(): {id: [], its continuous values, id: [], id: [], ...}
    :param feature_index: the index of selected features being applied.
    :param pair_ids: n data points
    :param interval_boundary_points:
    :param mean_matrix: m features * k intervals
    :param sigma_matrix: m features * k intervals
    :return: sim_X_mean: n data points * m means
              sim_X_variance: n data points * m variances
    """
    sim_X_mean = []
    sim_X_variance = []
    for _id in pair_ids:
        _feature_mean = []
        _feature_variance = []
        continue_values = id2contvalue.get(_id)
        for i in range(len(feature_index)):
            _feature_value = continue_values[feature_index[i]]
            _interval_index = get_interval_index(_feature_value, interval_boundary_points)
            # _feature_mean.append(_feature_value)
            if mean_matrix[i][_interval_index] != -1:
                _feature_mean.append(mean_matrix[i][_interval_index])
                _feature_variance.append(sigma_matrix[i][_interval_index])
            else:
                _feature_mean.append(0.0)
                _feature_variance.append(0.0)
        sim_X_mean.append(_feature_mean)
        sim_X_variance.append(_feature_variance)
    sim_X_mean = np.array(sim_X_mean)
    sim_X_variance = np.array(sim_X_variance)
    return sim_X_mean, sim_X_variance

# 2020-07-24
def get_mul_probability_input_X(id2contvalue, class_num, pair_ids, interval_boundary_points):
    _class_num = class_num
    _interval_num = len(interval_boundary_points) - 1

    _sparse_X = [[[0] * _interval_num for _ in range(class_num)] for _ in range(len(pair_ids))]
    _sparse_X_mean = [[[0] * _interval_num for _ in range(class_num)] for _ in range(len(pair_ids))]

    for k in range(len(pair_ids)):
        _pair_id = pair_ids[k]
        continue_values = id2contvalue.get(_pair_id)
        for i in range(class_num):
            _feature_value = continue_values[0][i]
            _interval_index = get_interval_index(_feature_value, interval_boundary_points)
            _sparse_X[k][i][_interval_index] = 1
            _sparse_X_mean[k][i][_interval_index] = _feature_value

    return _sparse_X_mean, _sparse_X


# # ????
# def get_mul_probability_input_X(id2contvalue, class_num, pair_ids, interval_boundary_points, mean_matrix, sigma_matrix):
#     _class_num = class_num
#     _interval_num = len(interval_boundary_points) - 1
#     _total_intervals = _class_num * _interval_num
#     # print(_class_num, _interval_num, _total_intervals)
#     '''
#     _sparse_X = sp.lil_matrix((len(pair_ids), _total_intervals))
#     _sparse_X_mean = sp.lil_matrix((len(pair_ids), _total_intervals))
#     _sparse_X_variance = sp.lil_matrix((len(pair_ids), _total_intervals))
#     '''
#     _sparse_X = [[[0] * _interval_num for _ in range(class_num)] for _ in range(len(pair_ids))]
#     _sparse_X_mean = [[[0] * _interval_num for _ in range(class_num)] for _ in range(len(pair_ids))]
#     _sparse_X_variance = [[[0] * _interval_num for _ in range(class_num)] for _ in range(len(pair_ids))]
#     # print(_sparse_X_variance.get_shape())
#     for k in range(len(pair_ids)):
#         _pair_id = pair_ids[k]
#         continue_values = id2contvalue.get(_pair_id)
#         for i in range(class_num):
#             _feature_value = continue_values[0][i]
#             _interval_index = get_interval_index(_feature_value, interval_boundary_points)
#             # print('--------------------------------_interval_index')
#             # print(_interval_index)
#             # index_offset = i * _interval_num + _interval_index
#             _sparse_X[k][i][_interval_index] = 1
#             # print(k, i, _interval_index)
#             _sparse_X_mean[k][i][_interval_index] = _feature_value
#             if mean_matrix[i][_interval_index] != -1:
#                 _sparse_X_variance[k][i][_interval_index] = sigma_matrix[i][_interval_index]
#             else:
#                 _sparse_X_variance[k][i][_interval_index] = 0.0
#     '''
#     print('------------------------------_sparse_X_mean')
#     print(_sparse_X_mean)
#     print('------------------------------_sparse_X_variance')
#     print(_sparse_X_variance)
#
#     print('-----------------------------_sparse_X')
#     print(_sparse_X)
#     '''
#     return _sparse_X_mean, _sparse_X_variance, _sparse_X


def get_probability_input_X(id2contvalue,
                            feature_index,
                            pair_ids,
                            interval_boundary_points,
                            mean_matrix,
                            sigma_matrix):
    """

    :param id2contvalue: dict(): {id: [], its continuous values, id: [], id: [], ...}
    :param feature_index: the index of selected features being applied.
    :param pair_ids: n data points
    :param interval_boundary_points:
    :param mean_matrix: m features * k intervals
    :param sigma_matrix: m features * k intervals
    :return: X_mean: n data points * (m * k) means
              X_variance: n data points * (m * k) variances
              X_activated_matrix:
    """

    _feature_num = len(feature_index)
    _interval_num = len(interval_boundary_points) - 1
    _total_intervals = _feature_num * _interval_num
    _sparse_X = sp.lil_matrix((len(pair_ids), _total_intervals))
    _sparse_X_mean = sp.lil_matrix((len(pair_ids), _total_intervals))
    _sparse_X_variance = sp.lil_matrix((len(pair_ids), _total_intervals))
    for k in range(len(pair_ids)):
        _pair_id = pair_ids[k]
        continue_values = id2contvalue.get(_pair_id)
        for i in range(len(feature_index)):
            _feature_value = continue_values[feature_index[i]]
            _interval_index = get_interval_index(_feature_value, interval_boundary_points)
            index_offset = i * _interval_num + _interval_index
            _sparse_X[k, index_offset] = 1
            _sparse_X_mean[k, index_offset] = _feature_value
            if mean_matrix[i][_interval_index] != -1:
                _sparse_X_variance[k, index_offset] = sigma_matrix[i][_interval_index]
            else:
                _sparse_X_variance[k, index_offset] = 0.0
    return _sparse_X_mean, _sparse_X_variance, _sparse_X
