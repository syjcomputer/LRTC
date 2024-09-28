import codecs
import os

import six
from torch.utils.data import Dataset
import json
import torch.nn as nn
import tensorflow as tf
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer


def load_data(filename):
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            D.append((l['text'], int(l['label'])))
    return D

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

class textDataset(Dataset):
    def __init__(self, datas, labels):
        super().__init__()
        self.texts = datas
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text,label

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight)
        #nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

######################################### 改写bert4keras相关函数用于train_keras.py
# def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
#     """从bert的词典文件中读取词典
#     """
#     token_dict = {}
#     with open(dict_path, encoding=encoding) as reader:
#         for line in reader:
#             token = line.split()
#             token = token[0] if token else line.strip()
#             token_dict[token] = len(token_dict)
#
#     if simplified:  # 过滤冗余部分token
#         new_token_dict, keep_tokens = {}, []
#         startswith = startswith or []
#         for t in startswith:
#             new_token_dict[t] = len(new_token_dict)
#             keep_tokens.append(token_dict[t])
#
#         for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
#             if t not in new_token_dict and not Tokenizer._is_redundant(t):
#                 new_token_dict[t] = len(new_token_dict)
#                 keep_tokens.append(token_dict[t])
#
#         return new_token_dict, keep_tokens
#     else:
#         return token_dict
#
#
# def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
#     """Numpy函数，将序列padding到同一长度
#     """
#     if length is None:
#         length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
#     elif not hasattr(length, '__getitem__'):
#         length = [length]
#
#     slices = [np.s_[:length[i]] for i in range(seq_dims)]
#     slices = tuple(slices) if len(slices) > 1 else slices[0]
#     pad_width = [(0, 0) for _ in np.shape(inputs[0])]
#
#     outputs = []
#     for x in inputs:
#         x = x[slices]
#         for i in range(seq_dims):
#             if mode == 'post':
#                 pad_width[i] = (0, length[i] - np.shape(x)[i])
#             elif mode == 'pre':
#                 pad_width[i] = (length[i] - np.shape(x)[i], 0)
#             else:
#                 raise ValueError('"mode" argument must be "post" or "pre".')
#         x = np.pad(x, pad_width, 'constant', constant_values=value)
#         outputs.append(x)
#
#     return np.array(outputs)
#
# is_py2 = six.PY2
#
# if not is_py2:
#     basestring = str
#
# def is_string(s):
#     """判断是否是字符串
#     """
#     return isinstance(s, basestring)
#
# class DataGenerator(object):
#     """数据生成器模版
#     """
#     def __init__(self, data, batch_size=32, buffer_size=None):
#         self.data = data
#         self.batch_size = batch_size
#         if hasattr(self.data, '__len__'):
#             self.steps = len(self.data) // self.batch_size
#             if len(self.data) % self.batch_size != 0:
#                 self.steps += 1
#         else:
#             self.steps = None
#         self.buffer_size = buffer_size or batch_size * 1000
#
#     def __len__(self):
#         return self.steps
#
#     def sample(self, random=False):
#         """采样函数，每个样本同时返回一个is_end标记
#         """
#         if random:
#             if self.steps is None:
#
#                 def generator():
#                     caches, isfull = [], False
#                     for d in self.data:
#                         caches.append(d)
#                         if isfull:
#                             i = np.random.randint(len(caches))
#                             yield caches.pop(i)
#                         elif len(caches) == self.buffer_size:
#                             isfull = True
#                     while caches:
#                         i = np.random.randint(len(caches))
#                         yield caches.pop(i)
#
#             else:
#
#                 def generator():
#                     for i in np.random.permutation(len(self.data)):
#                         yield self.data[i]
#
#             data = generator()
#         else:
#             data = iter(self.data)
#
#         d_current = next(data)
#         for d_next in data:
#             yield False, d_current
#             d_current = d_next
#
#         yield True, d_current
#
#     def __iter__(self, random=False):
#         raise NotImplementedError
#
#     def forfit(self, random=True):
#         while True:
#             for d in self.__iter__(random):
#                 yield d
#
#     def fortest(self, random=False):
#         while True:
#             for d in self.__iter__(random):
#                 yield d[0]
#
#     def to_dataset(self, types, shapes, names=None, padded_batch=False):
#         """转为tf.data.Dataset格式
#         如果传入names的话，自动把数据包装成dict形式。
#         """
#         if names is None:
#
#             generator = self.forfit
#
#         else:
#
#             if is_string(names):
#                 warps = lambda k, v: {k: v}
#             elif is_string(names[0]):
#                 warps = lambda k, v: dict(zip(k, v))
#             else:
#                 warps = lambda k, v: tuple(
#                     dict(zip(i, j)) for i, j in zip(k, v)
#                 )
#
#             def generator():
#                 for d in self.forfit():
#                     yield warps(names, d)
#
#             types = warps(names, types)
#             shapes = warps(names, shapes)
#
#         if padded_batch:
#             dataset = tf.data.Dataset.from_generator(
#                 generator, output_types=types
#             )
#             dataset = dataset.padded_batch(self.batch_size, shapes)
#         else:
#             dataset = tf.data.Dataset.from_generator(
#                 generator, output_types=types, output_shapes=shapes
#             )
#             dataset = dataset.batch(self.batch_size)
#
#         return dataset

############################3 keras_train

def get_token_dict(dict_path):
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


class MyTokenizer(Tokenizer):

    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            # elif self._is_space(c):
            #     R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class Keras_DataGenerator:

    def __init__(self, data, tokenizer, batch_size=64, maxlen=128):
        '''

        :param data: [(text,label)]
        :param tokenizer:
        :param batch_size:
        :param maxlen:
        '''
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:self.maxlen]
                x1, x2 = self.tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                # tmp = w
                Y.append([y])   # 要使用独热编码
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []