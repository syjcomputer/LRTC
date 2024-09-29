from tqdm import tqdm
import os
import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
# from model import *
# from Resnet import *
from torch.nn.functional import softmax
import json
import tensorflow as tf
from keras.models import Model
from keras.models import load_model

from bert4keras.snippets import sequence_padding, DataGenerator
from torch.utils.data import TensorDataset, DataLoader

from RiskAnalysis.common.config import cfg

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_data1(filename):
    text = []
    label = []
    id = []
    with open(filename, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            text.append(l['text'])
            label.append(int(l['label']))
            id.append(l['id'])
    return text, label, id


def load_data(texts, labels):
    D = []
    for i in range(len(labels)):
        D.append((texts[i], int(labels[i][0])))
    return D

def load_data2(texts, labels):
    D = []
    for i in range(len(labels)):
        D.append((texts[i], int(labels[i])))
    return D

def load_text(name):
    texts = []
    data_texts, data_labels, data_ids = load_data1(os.path.join(cfg.get_risk_dataset_path() + '/' + name + '.json'))
    for i in range(len(data_texts)):
        text = data_texts[i]
        texts.append(text)
    return texts

import multiprocessing

def multi_load_data(texts, labels, tokenizer, maxlen):
    token_ids_list = []
    segment_ids_list = []
    for one_text in texts:
        token_ids, segment_ids = tokenizer.encode(one_text, maxlen=maxlen)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)
    token_ids_list = sequence_padding(token_ids_list)
    segment_ids_list = sequence_padding(segment_ids_list)
    return [token_ids_list, segment_ids_list, np.array(labels).reshape(-1, 1), np.array(texts)]


def predict_on_batch(model, token_ids, segment_ids, dense_model):
    predictions = model.predict([token_ids, segment_ids])
    dense_vector = dense_model([tf.convert_to_tensor(token_ids), tf.convert_to_tensor(segment_ids)])
    return dense_vector, predictions.argmax(-1)


def predict(data_gen, model, dense_model):
    denses = []
    predictions = []
    labels = []
    texts = []

    total = 0
    correct = 0
    for token_ids, segment_ids, label_batch, texts_batch in tqdm(data_gen):
        # token_ids = data_batch[0]
        # segment_ids = data_batch[1]
        batch_dense, batch_predictions = predict_on_batch(model, token_ids, segment_ids, dense_model)
        predictions.extend(batch_predictions)
        denses.extend(batch_dense)
        labels.extend(label_batch)
        texts.extend(texts_batch)

        for count in range(len(label_batch)):
            if int(batch_predictions[count]) == int(label_batch[count]):
                correct += 1
            total += 1

    acc = 100. * float(correct) / total
    return predictions, denses, labels, texts, acc

def base_model_test(model, tokenizer, name, batch_size=64, maxlen=150):

    critierion = nn.CrossEntropyLoss()

    data_texts, data_labels, data_ids = load_data1(
        os.path.join(cfg.get_finetune_path() + '/' + name + '.json'))  # json是原数据， csv是风险模型数据

    layer_name = 'Transformer-11-FeedForward-Norm'
    bert_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    layer_name1 = 'classify_output'
    dense_model = Model(inputs=model.input, outputs=model.get_layer(layer_name1).output)

    distribution_dense = []

    targets = []

    predictions = []

    texts = []

    correct = 0
    total = 0

    ################################################## batch train(keras)
    for i in tqdm(range(0, len(data_texts), batch_size)):
        text = np.array(data_texts)[i: i+batch_size]
        label = np.array(data_labels)[i: i+batch_size]

        token_ids_list = []
        segment_ids_list = []
        for one_text in text:
            token_ids, segment_ids = tokenizer.encode(one_text, maxlen=maxlen)
            token_ids_list.append(token_ids)
            segment_ids_list.append(segment_ids)

        # bert_vector = bert_model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
        token_ids_list = sequence_padding(token_ids_list)
        segment_ids_list = sequence_padding(segment_ids_list)

        dense_vector = dense_model.predict([token_ids_list,segment_ids_list])

        pred = model.predict([token_ids_list,segment_ids_list]).argmax(axis=-1)

        distribution_dense.extend(dense_vector.tolist())    #  (all_data, class_nums)
        targets.extend(label.reshape((-1,1)).tolist())  # (all_data, 1)
        predictions.extend(pred.tolist())   # (all_data,1)
        texts.extend(text.tolist())

        # loss
        t1 = torch.tensor(distribution_dense)
        t2 = torch.tensor(targets).squeeze()
        loss = critierion(t1,t2)


        # ids.append([id])
        for count in range(len(label)):
            if int(pred[count]) == int(label[count]):
                correct += 1
            total += 1

    acc = 100. * float(correct) / total
    return distribution_dense, targets, predictions, texts, acc, loss.item()


class data_generator(DataGenerator):
    """
    """

    def __init__(self, data, tokenizer, maxlen, batch_size=32, buffer_size=None):
        DataGenerator.__init__(self, data, batch_size, buffer_size)
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, num) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.maxlen)
            labels = [0] * len(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([num])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
