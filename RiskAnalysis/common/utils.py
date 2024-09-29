import numpy as np
from RiskAnalysis.common.config import cfg
import os


def get_predict_label(_probs):
    '''
    :param _probs: the prob of classifier
    :return:
    get machine label for 2-classes
    '''
    prob_temp = np.array(_probs)
    _label = (prob_temp >= 0.5).astype(int)
    return _label


def get_predict_labels(_id):
    '''
    :param _id:
    :return:
    get machine label for mul-classes, which need a file about machine label
    '''
    labels = []
    import pandas as pd
    label = pd.read_csv(os.path.join(cfg.risk_dataset_path, 'machine_labels.csv'))
    label_ = {}
    for i in range(label.shape[0]):
        label_[label[i:i + 1]['id'].values[0]] = label[i:i + 1]['pre_label'].values[0]
    for i in range(len(_id)):
        labels.append(label_[_id[i]])
    labels = np.array(labels)
    return labels


def get_true_label(_ids, id_2_pinfo):
    _y = []
    for _id in _ids:
        _y.append(id_2_pinfo.get(_id)[1])
    return _y


def calculate_rules_feature_mu_sigma(ob_ids, matrix, label, ob_labels):
    _labels = []
    i = 0
    for _id in ob_ids:
        if (matrix[i] == 0):
            i += 1
            continue
        if ob_labels.get(_id) == label:
            _labels.append(1)
        else:
            _labels.append(0)
        i += 1

    _labels = np.array(_labels)
    _mu = np.average(_labels)
    _delta = (_labels - _mu) ** 2
    _sum = np.sum(_delta)
    _sigma = _sum / np.maximum(len(_labels) - 1, 1)
    return [_mu, _sigma]


def calculate_machine_feature_mu_sigma(ob_ids, ob_labels, class_id):
    _labels = []
    class_id = int(class_id[1:])
    for _id in ob_ids:
        if ob_labels.get(_id) == class_id:
            _labels.append(1)
        else:
            _labels.append(0)

    _labels = np.array(_labels)
    _mu = np.average(_labels)
    _delta = (_labels - _mu) ** 2
    _sum = np.sum(_delta)
    _sigma = _sum / np.maximum(len(_labels) - 1, 1)
    # - Select the discriminative features.
    # if 0.1 < _mu < 0.9:
    #     return None
    # else:
    #     return [_mu, _sigma]
    # - No selection.
    return [_mu, _sigma]


# def calculate_feature_mu_sigma(ob_ids, ob_labels, class_id='M1', risk_labels=None):  # M:rule C:class
#     """
#     :param ob_ids: observed data ids.
#     :param ob_labels: observed data labels.
#     :return: _mu: mean (sample mean)
#               _sigma: variance (the second central moment of samples)
#
#     2020.04 calculate mu_sigma of rules feature and machine feature
#     2020.05 obsolete
#     """
#     _labels = []
#     if class_id == 'C':
#         for _id in ob_ids:
#             print(_id, risk_labels.get(_id), ob_labels.get(_id))
#             if risk_labels.get(_id) == 0:
#                 _labels.append(1)
#             else:
#                 _labels.append(0)
#     else:
#         class_id = int(class_id[1:])
#
#         for _id in ob_ids:
#             if ob_labels.get(_id) == class_id:
#                 _labels.append(1)
#             else:
#                 _labels.append(0)
#     _labels = np.array(_labels)
#     _mu = np.average(_labels)
#     _delta = (_labels - _mu) ** 2
#     _sum = np.sum(_delta)
#     _sigma = _sum / np.maximum(len(_labels) - 1, 1)
#     # - Select the discriminative features.
#     # if 0.1 < _mu < 0.9:
#     #     return None
#     # else:
#     #     return [_mu, _sigma]
#     # - No selection.
#     return [_mu, _sigma]
