
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

# from train import data_generator
import csv
import jieba
import torchtext
from torchtext.data import Field,TabularDataset,Iterator,BucketIterator,LabelField
import pandas as pd

import torch as th
# import dgl
import torch.utils.data as Data
import scipy.sparse as sp
#from base_models.BertGCN_utils.train_bert_gcn import train_process, encode_input
# from base_models.BertGCN import BertGCN
# from base_models.BertGCN_utils.build_graph import load_corpus, normalize_adj
# from transformers import AutoModelForSequenceClassification, AutoConfig, BertModel

from get_CNN_feature import TextCNN,stopwordslist,get_params,torchtext_file

from utils.train_util import load_data1, load_data #, get_token_dict, Keras_DataGenerator
from configs.get_feature_vector_config import *
# from train_pytorch import get_dataloader
# from train_keras import build_bert , MyTokenizer
import keras
# from transformers import BertTokenizer, AutoModelForSequenceClassification
# from keras_bert import load_trained_model_from_checkpoint

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

    cur_path = '{}_bert/{}/'.format(args.data_name, args.rate) # train path
    fine_tune_path = '{}_bert/{}/'.format(fine_tune_dataset, args.rate) # fine-tune path

    # convert files
    # old_files = os.path.join(cur_path, "train.json")
    # new_files = os.path.join(cur_path, "new_train.json")
    # torchtext_file(old_files, new_files)

    nameList = ["train", "test", "val"]
    for name in nameList:
        old_files = os.path.join(fine_tune_path, f"{name}.json")
        new_files = os.path.join(fine_tune_path, f"new_{name}.json")
        torchtext_file(old_files, new_files)

    def cut(sentence):
        return [token for token in jieba.lcut(sentence) if token not in stopwords]
    # 声明一个Field对象，对象里面填的就是需要对文本进行哪些操作，比如这里lower=True英文大写转小写,tokenize=cut对于文本分词采用之前定义好的cut函数，sequence=True表示输入的是一个sequence类型的数据，还有其他更多操作可以参考文档
    TEXT = Field(sequential=True, lower=True, tokenize=cut)
    # 声明一个标签的LabelField对象，sequential=False表示标签不是sequence，dtype=torch.int64标签转化成整形
    LABEL = LabelField(sequential=False, use_vocab=False)
    ID = Field(sequential=False, use_vocab=False)

    # 这里主要是告诉torchtext需要处理哪些数据，这些数据存放在哪里，TabularDataset是一个处理scv/tsv的常用类
    train_dataset, dev_dataset, test_dataset = TabularDataset.splits(
        path=f'{fine_tune_dataset}_bert/{args.rate}/',  # 文件存放路径
        #path=f'BBC_bert/',  # 文件存放路径
        format='json',  # 文件格式
        skip_header=False,  # 是否跳过表头，我这里数据集中没有表头，所以不跳过
        train='new_train.json',
        validation='new_val.json',
        test='new_test.json',
        #fields=[('id', None), ('label', LABEL), ('text', TEXT)]
        fields={"id":("id",ID),'label':("label",LABEL), 'text': ("text",TEXT)}  # 定义数据对应的表头
    )

    vector_cache = "pretrained_vector"
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

    # 统一句子长度
    max_length = 150  # 假设统一为长度50

    # 将句子长度填充到最大长度
    # 将句子长度填充到最大长度
    # train_dataset = [(torch.tensor(TEXT.process([example.text]), dtype=torch.long), example.label) for example in
    #               train_dataset]
    # dev_dataset = [(torch.tensor(TEXT.process([example.text]), dtype=torch.long), example.label) for example in dev_dataset]
    # test_dataset = [(torch.tensor(TEXT.process([example.text]), dtype=torch.long), example.label) for example in test_dataset]
    # train_dataset = [(torch.tensor(example.text[:max_length]), example.label) for example in train_dataset]
    # dev_dataset = [(torch.tensor(example.text[:max_length]), example.label) for example in dev_dataset]
    # test_dataset = [(torch.tensor(example.text[:max_length]), example.label) for example in test_dataset]

    # args.vocab_size = len(TEXT.vocab)
    # args.embedding_dim = TEXT.vocab.vectors.size()[-1] # 词向量维度
    # args.vector = TEXT.vocab.vectors # 词向量
    # vectors = TEXT.vocab.vectors
    #
    # model = TextCNN(args, vectors)
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
    # cur = cur_path + 'feature/'

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
        # checkpoint = th.load(cur_path + best_model)

        if model_name == "textcnn":
            model_path = f"{dataset}_cnn/{rate}/{best_model}"
            print(model_path)
            print(os.path.exists(model_path))
            checkpoint = th.load(model_path)
            args = get_params()

            model, train_iter, val_iter, test_iter = load_textcnn(args,checkpoint)

            if torch.cuda.is_available():
                model = model.to(device=gpu)
                #model.cuda()


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
        # elif model_name == "bert":
        #
        #     # if torch.cuda.is_available():
        #     #     device = torch.cuda()
        #
        #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #
        #     # model_path = f"{dataset}_bert/{rate}/{best_model}"
        #
        #     model_path = f"{dataset}_bert/{rate}/"
        #
        #     model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels = classify_num_labels)
        #
        #     tokenizer = BertTokenizer.from_pretrained(f'{dict_path}/')
        #
        #     data_texts, data_labels, data_ids = load_data1(data_path + name + '.json')
        #
        #     dataloader = get_dataloader2(tokenizer, data_ids, data_texts, data_labels, maxlen, batch_size)
        #
        #     model = model.to(device)
        #
        #     with torch.no_grad():
        #         model.eval()
        #         for i, data in tqdm(enumerate(dataloader)):
        #             input_ids, attention_mask, y, id = data
        #
        #             input_ids = input_ids.to(device)
        #             attention_mask = attention_mask.to(device)
        #             y = y.to(device)
        #
        #             out = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=y, output_hidden_states=True)
        #
        #             # 获取各层
        #             pred_labels = out.logits.argmax(dim=1)
        #             # pool_output = out.pooler_output
        #             logits = out.logits
        #             last_hidden_state = out.hidden_states[-1]
        #             pool_output = torch.mean(last_hidden_state, dim=1)
        #
        #             #
        #             # print(len(pool_output))
        #             # print(len(logits))
        #             # print(len(id))
        #             distribution_bert.extend(pool_output.detach().cpu().numpy().tolist())
        #             distribution_dense.extend(logits.detach().cpu().numpy().tolist())
        #             targets.extend(y.reshape(-1,1).detach().cpu().numpy().tolist())
        #             predictions.extend(pred_labels.reshape(-1,1).detach().cpu().numpy().tolist())
        #
        #             id = id.detach().cpu().numpy().tolist()
        #             new_id = [name+"/"+str(i) for i in id]
        #             ids.extend(new_id)
        #         path = f"{fine_tune_dataset}_bert/{rate}/"
        #         save_feature_vector(path, name, distribution_bert, distribution_dense, targets, predictions, ids)

    # elif model_name == "keras_bert":
    #
    #     model = build_bert()
    #
    #     model.summary()
    #
    #     tokenizer = MyTokenizer(get_token_dict(dict_path))
    #
    #     data_texts, data_labels, data_ids = load_data1(data_path + name + '.json')
    #
    #     cur_path = '{}/{}/'.format(data_name, rate)
    #     model_path = cur_path + 'best_model.weights'
    #     model.load_weights(model_path)
    #     # model = load_model(cur_path + best_model)
    #     print("load model successfully")
    #
    #     """
    #     Pooler-Dense (Dense)            (None, 768)
    #     final_Dropout (None, 768)
    #     Transformer-11-FeedForward-Norm (None, None, 768)
    #     """
    #     dense_layer = model.get_layer('classify_output').output
    #     dense_model = Model(model.inputs, dense_layer)
    #
    #     bert_model = model.layers[2]
    #     bert_model.summary()
    #
    #     pool_output = bert_model.layers[-6].output
    #     pool_extractor = keras.Model(bert_model.inputs, outputs=pool_output)
    #
    #     print("len(data_texts)", len(data_texts))
    #     print("len(data_labels)", len(data_labels))
    #     print("len(data_ids)", len(data_ids))
    #     print()
    #
    #     for i in tqdm(range(len(data_texts)), desc="calculate"):
    #         text = data_texts[i]
    #         label = data_labels[i]
    #         id = name + '/' + str(data_ids[i])
    #         token_ids, segment_ids = tokenizer.encode(first=text)
    #         # token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    #
    #         bert_vector = pool_extractor.predict([np.array([token_ids]), np.array([segment_ids])])[0][0]
    #         # bert_vector = extract_embeddings(model, [(token_ids, segment_ids)])[0][0]
    #
    #         dense_vector = dense_model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
    #
    #         pred = model.predict([np.array([token_ids]), np.array([segment_ids])]).argmax(axis=1)
    #
    #         #
    #         distribution_bert.append(bert_vector)
    #         distribution_dense.append(dense_vector)
    #         targets.append([label])
    #         predictions.append(list(pred))
    #         ids.append([id])
    #
    #     save_path = f"{fine_tune_dataset}_bert/{rate}/"
    #     save_feature_vector(save_path, name, distribution_bert, distribution_dense, targets, predictions, ids)


        # if model_name == 'BertGCN':
        #     # from train_bert_gcn import set_args
        #
        #     # args = set_args()
        #
        #     # model = BertGCN(nb_class=20, pretrained_model=args.bert_init, m=args.m, gcn_layers=args.gcn_layers,
        #     #                n_hidden=args.n_hidden, dropout=args.dropout)
        #     # model.bert_model.load_state_dict(checkpoint[get_layer])
        #     # model.bert_model.load_state_dict(checkpoint[get_layer1])
        #
        #     # g_path = f'base_models/BertGCN_utils/{data_name}/{rate}/bert.g'
        #
        #     # if os.path.exists(g_path):
        #
        #     train_data = load_data1(data_path + 'train.json')  # [[texts], [labels], [ids]]
        #
        #     valid_data = load_data1(data_path + 'val.json')
        #
        #     test_data = load_data1(data_path + 'test.json')
        #
        #     ###################################### loader不必要，需要ids
        #
        #     model, g, texts, labels, all_id, idx_loader_train, idx_loader_val, idx_loader_test, idx_loader = train_process(
        #         data_name, rate, train_data, valid_data, test_data, True, cur_path + best_model)
        #
        #     # set hook
        #     # model.bert_model.register_forward_hook(hook)
        #     # model.classifier.register_forward_hook(hook)
        #
        #     model = model.to(gpu)
        #     g = g.to(gpu)
        #
        #     with torch.no_grad():
        #         for name in ['train', 'val', 'test']:
        #             distribution_bert = []
        #
        #             distribution_dense = []
        #
        #             targets = []
        #
        #             predictions = []
        #
        #             ids = []
        #             #####################################################3
        #             i = 0
        #             for batch in eval('idx_loader_' + name):
        #                 (idx,) = [x.to(gpu) for x in batch]  # idx: (batch_size, )
        #                 mask = g.ndata[name][idx].type(th.BoolTensor)
        #
        #                 value1 = SaveValues(model.bert_model)
        #                 value2 = SaveValues(model.classifier)
        #
        #                 y_pred = model(g, idx)[mask].argmax(axis=1)  # (batch_size, )
        #                 y_true = g.ndata['label_train'][idx][mask]  # (batch_size, )
        #                 bert_vector = value1.features_out_hook[0][:, 0]  # (batch_size, feat_nums)
        #                 dense_vector = value2.features_out_hook  # (batch_size, class_nums)
        #
        #                 y_true = np.array(y_true.cpu()).reshape(-1, 1).tolist()
        #                 y_pred = np.array(y_pred.cpu()).reshape(-1, 1).tolist()
        #                 idx = np.array(all_id)[i:i + len(idx)].reshape(-1, 1).tolist()
        #                 idx = [name + '/' + str(id) for id in idx]
        #                 i += len(idx)
        #
        #                 #
        #                 # print(bert_vector.cpu())
        #                 # print(dense_vector.cpu())
        #                 # print(len(bert_vector))
        #
        #                 distribution_bert.extend(bert_vector.cpu().numpy())
        #                 distribution_dense.extend(dense_vector.cpu().numpy())
        #                 targets.extend(y_true)
        #                 predictions.extend(y_pred)
        #                 ids.extend(idx)
        #
        #                 # print(distribution_bert)
        #
        #             pd.DataFrame(distribution_bert).to_csv(cur_path + 'distribution_net_' + name + '.csv',
        #                                                    header=None, index=None)
        #             pd.DataFrame(distribution_dense).to_csv(cur_path + 'distribution_dense_' + name + '.csv',
        #                                                     header=None, index=None)
        #             pd.DataFrame(targets).to_csv(cur_path + 'targets_' + name + '.csv', header=None, index=None)
        #             pd.DataFrame(predictions).to_csv(cur_path + 'predictions_' + name + '.csv', header=None,
        #                                              index=None)
        #             pd.DataFrame(ids).to_csv(cur_path + 'ids_' + name + '.csv', header=None, index=None)
        #
        #             np.save(cur_path + 'distribution_net_' + name + '.npy', distribution_bert)
        #             np.save(cur_path + 'distribution_dense_' + name + '.npy', distribution_dense)
        #             np.save(cur_path + 'targets_' + name + '.npy', targets)
        #             np.save(cur_path + 'predictions_' + name + '.npy', predictions)
        #             np.save(cur_path + 'ids_' + name + '.npy', ids)


if __name__ == '__main__':
    print("--------------------------- start extract features")
    if model_name == 'BertGCN':
        '''if bertgcn, you need set args in train_bert_gcn.py'''
        get_feature(None)
        print('------------------------------------ over')
    elif model_name=="bert" or model_name=="keras_bert":
        print("bert")
        get_feature('train')
        get_feature('val')
        get_feature('test')
        print('--------------------------------------- over!!!')
    elif model_name=="textcnn":
        get_feature(None)
        print('--------------------------------------- over!!!')
