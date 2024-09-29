
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# zh
import os
import json
import numpy as np
import torch
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import open
from keras import Input
from keras.models import Model
from keras.layers import Dropout, Dense, Lambda
from bert4keras.models import build_transformer_model
from bert4keras.backend import keras
from keras.models import load_model
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import csv
import jieba
import torchtext
from torchtext.data import Field,TabularDataset,BucketIterator,LabelField
import pandas as pd

import torch as th
import torch.utils.data as Data
import scipy.sparse as sp

from PrepareRiskDataset.train_textcnn import TextCNN,stopwordslist,torchtext_file,get_params

from PrepareRiskDataset.train_util import load_data1, load_data #, get_token_dict, Keras_DataGenerator
from PrepareRiskDataset.config.get_feature_vector_config import *
import keras

class SaveValues():
    def __init__(self, layer):
        self.features_out_hook = None
        self.module_name = None
        self.features_in_hook = None

        # set hook
        self.forward_hook = layer.register_forward_hook(self.hook)

    def hook(self, module, fea_in, fea_out):
        # print("hooker working")
        self.module_name = module.__class__
        self.features_in_hook = fea_in
        self.features_out_hook = fea_out
        # return None


def load_textcnn(args, checkpoint):
    if args.data_name in ["fudan", "qinghua"]:
        print("CHA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        stopwords = stopwordslist("./stopwords/cn_stopwords.txt")
    else:
        print("ENG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        stopwords = stopwordslist("./stopwords/en.txt")

    # fine_tune_path = '{}_bert/{}/'.format(fine_tune_dataset, args.rate) # fine-tune path
    # nameList = ["train", "test", "val"]
    # for name in nameList:
    #     old_files = os.path.join(fine_tune_path, f"{name}.json")
    #     new_files = os.path.join(fine_tune_path, f"new_{name}.json")
    #     torchtext_file(old_files, new_files)

    def cut(sentence):
        return [token for token in jieba.lcut(sentence) if token not in stopwords]
    # 声明一个Field对象，对象里面填的就是需要对文本进行哪些操作，比如这里lower=True英文大写转小写,tokenize=cut对于文本分词采用之前定义好的cut函数，sequence=True表示输入的是一个sequence类型的数据，还有其他更多操作可以参考文档
    TEXT = Field(sequential=True, lower=True, tokenize=cut)
    # 声明一个标签的LabelField对象，sequential=False表示标签不是sequence，dtype=torch.int64标签转化成整形
    LABEL = LabelField(sequential=False, use_vocab=False)
    ID = Field(sequential=False, use_vocab=False)

    # 这里主要是告诉torchtext需要处理哪些数据，这些数据存放在哪里，TabularDataset是一个处理scv/tsv的常用类
    train_dataset, dev_dataset, test_dataset = TabularDataset.splits(
        path=f'{source_dataset}_bert/{args.rate}/',  # 文件存放路径
        format='json',  # 文件格式
        skip_header=False,  # 是否跳过表头，我这里数据集中没有表头，所以不跳过
        train='new_train.json',
        validation='new_val.json',
        test='new_test.json',
        #fields=[('id', None), ('label', LABEL), ('text', TEXT)]
        fields={"id":("id",ID),'label':("label",LABEL), 'text': ("text",TEXT)}  # 定义数据对应的表头
    )

    # vector_cache = "pretrained_vector"
    if not os.path.exists(vector_cache):
        os.mkdir(vector_cache)
    pretrained_name = 'crawl-300d-2M.vec'  # 预训练词向量文件名
    #pretrained_path = './drive/My Drive/TextCNN/word_embedding'  # 预训练词向量存放路径
    vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=vector_cache)
    args.vocab_size = vectors.vectors.size(0)
    args.embedding_dim = vectors.vectors.size(1) # 词向量维度

    TEXT.build_vocab(train_dataset, dev_dataset, test_dataset,
                     vectors=vectors)
    LABEL.build_vocab(train_dataset, dev_dataset, test_dataset)
    vector = vectors.vectors
    model = TextCNN(args, vector)

    model.load_state_dict(checkpoint)

    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset),  # 需要生成迭代器的数据集
        batch_sizes=(64, 64, 64),  # 每个迭代器分别以多少样本为一个batch
        sort_key=lambda x: len(x.text)  # 按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
    )
    return model, train_iter, dev_iter, test_iter

def save_feature_vector(path, name,distribution_bert,distribution_dense,targets,predictions,ids):
    if os.path.exists(path)==False:
        os.makedirs(path)

    pd.DataFrame(distribution_bert).to_csv(path + 'distribution_net_' + name + '.csv', header=None, index=None)
    pd.DataFrame(distribution_dense).to_csv(path + 'distribution_dense_' + name + '.csv', header=None,
                                            index=None)
    pd.DataFrame(targets).to_csv(path + 'targets_' + name + '.csv', header=None, index=None)
    pd.DataFrame(predictions).to_csv(path + 'predictions_' + name + '.csv', header=None, index=None)
    pd.DataFrame(ids).to_csv(path + 'ids_' + name + '.csv', header=None, index=None)

    np.save(path + 'distribution_net_' + name + '.npy', distribution_bert)
    np.save(path + 'distribution_dense_' + name + '.npy', distribution_dense)
    np.save(path + 'targets_' + name + '.npy', targets)
    np.save(path + 'predictions_' + name + '.npy', predictions)
    np.save(path + 'ids_' + name + '.npy', ids)

    print(f"save all files in {path}")

def get_dataloader2(tokenizer, id, text, label, maxlen, batch_size):
    input_labels = torch.unsqueeze(torch.tensor(label),dim=1)
    input_ids = torch.unsqueeze(torch.tensor(id),dim=1)
    res = tokenizer(text, padding=True,truncation=True,max_length=maxlen)
    data_set=TensorDataset(torch.LongTensor(res['input_ids']),
                           torch.LongTensor(res['attention_mask']),
                           torch.LongTensor(input_labels),
                           torch.LongTensor(input_ids))
    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             shuffle=True)
    return data_loader


def get_feature(name):

    distribution_bert = []

    distribution_dense = []

    targets = []

    predictions = []

    ids = []

    if code_frame=="bert4keras" and model_name =="bert":
        tokenizer = Tokenizer(dict_path, do_lower_case=True)

        data_texts, data_labels, data_ids = load_data1(data_path + name + '.json')

        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_pool=True,
            return_keras_model=False,
        )

        classify_output = Dropout(rate=0.5, name='final_Dropout')(bert.model.output)
        classify_output = Dense(units=classify_num_labels,  # units是输出层维度
                                activation='softmax',
                                name='classify_output',
                                kernel_initializer=bert.initializer
                                )(classify_output)
        model = keras.models.Model(bert.model.input, classify_output)
        cur_path = '{}/{}/'.format(data_name, rate)
        model_path = cur_path + 'best_model.weights'
        model.load_weights(model_path)
        # model = load_model(cur_path + best_model)
        print("load model successfully")

        """
        Pooler-Dense (Dense)            (None, 768)
        final_Dropout (None, 768)
        Transformer-11-FeedForward-Norm (None, None, 768)
        """
        # bert
        # get_layer = 'Pooler-Dense'
        # get_layer1 = 'classify_output'

        layer_name = get_layer
        bert_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        layer_name1 = get_layer1
        dense_model = Model(inputs=model.input, outputs=model.get_layer(layer_name1).output)

        print("len(data_texts)", len(data_texts))
        print("len(data_labels)", len(data_labels))
        print("len(data_ids)", len(data_ids))
        print()

        for i in tqdm(range(len(data_texts)), desc="calculate"):
            text = data_texts[i]
            label = data_labels[i]
            id = name + '/' + str(data_ids[i])
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

            # # bert_vector = bert_model.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
            bert_vector = bert_model.predict([np.array([token_ids]), np.array([segment_ids])])[0]

            dense_vector = dense_model.predict([np.array([token_ids]), np.array([segment_ids])])[0]

            pred = model.predict([np.array([token_ids]), np.array([segment_ids])]).argmax(axis=1)

            #
            distribution_bert.append(bert_vector)
            distribution_dense.append(dense_vector)
            targets.append([label])
            predictions.append(list(pred))
            ids.append([id])

        save_path = f"{fine_tune_dataset}_bert/{rate}/"
        save_feature_vector(save_path, name, distribution_bert, distribution_dense, targets, predictions, ids)

    elif code_frame=="pytorch":

        if model_name == "textcnn":
            model_path = f"{dataset}_cnn/{rate}/{best_model}"
            print(f"load model from {model_path}")
            checkpoint = th.load(model_path)
            args = get_params()

            model, train_iter, val_iter, test_iter = load_textcnn(args,checkpoint)

            if torch.cuda.is_available():
                model = model.to(device=gpu)

            iters = [train_iter, val_iter, test_iter]
            names = ['train', 'val', 'test']
            #names = ['test']
            for i in range(len(names)):

                distribution_bert = []

                distribution_dense = []

                targets = []

                predictions = []

                ids = []

                iter = iters[i]

                name = names[i]

                for batch in tqdm(iter, desc=f"{name} batches"):
                    id = batch.id
                    feature, target = batch.text, batch.label
                    if torch.cuda.is_available():  # 如果有GPU将特征更新放在GPU上
                        # feature2, target2 = feature.cuda(), target.cuda()
                        feature2, target2 = feature.to(gpu), target.to(gpu)
                    logits, pools, dense = model(feature2)

                    preds = torch.max(logits, 1)[1].view(
                    target.size(), 1).cpu().numpy()

                    target = target.unsqueeze(1).numpy()
                    id = id.unsqueeze(1).numpy().tolist()

                    new_id = [name+"/"+str(i) for i in id]

                    distribution_bert.extend(pools.detach().cpu().numpy().tolist())
                    distribution_dense.extend(logits.detach().cpu().numpy().tolist())
                    targets.extend(target.tolist())
                    predictions.extend(preds.tolist())
                    ids.extend(new_id)
                path = f"{fine_tune_dataset}_cnn/{rate}/"
                save_feature_vector(path, name, distribution_bert, distribution_dense, targets, predictions, ids)


if __name__ == '__main__':
    print(f"--------------------------- start extract risk metrics with {model_name}--------------")
    if model_name=="bert":
        get_feature('train')
        get_feature('val')
        get_feature('test')
        print('--------------------------------------- over!!!')
    elif model_name=="textcnn":
        get_feature(None)
        print('--------------------------------------- over!!!')
