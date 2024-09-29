import numpy as np


def get_predict_label(_probs):
    prob_temp = np.array(_probs)
    _label = (prob_temp >= 0.5).astype(int)
    return _label


def get_true_label(_ids, id_2_pinfo):
    _y = []
    for _id in _ids:
        _y.append(id_2_pinfo.get(_id)[1])
    return _y


def calculate_feature_mu_sigma(ob_ids, ob_labels):
    """

    :param ob_ids: observed data ids.
    :param ob_labels: observed data labels.
    :return: _mu: mean (sample mean)
              _sigma: variance (the second central moment of samples)
    """
    _labels = []
    for _id in ob_ids:
        _labels.append(ob_labels.get(_id))
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
