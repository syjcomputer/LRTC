# coding=utf-8
import math
import json
import jieba
from copy import deepcopy
from sklearn.svm import SVC # 支持向量机模块
import pickle
import pandas as pd
from os.path import join
import numpy as np
# import matplotlib.pyplot as plt
# import umap
from queue import Queue,PriorityQueue
from collections import Counter
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from prepareRiskDataset.config.overall_config import *

data_name = f'{source_dataset}_bert'
cur_path = './{}/{}/'.format(data_name, rate)
fine_tune_path = f"{target_dataset}_bert/{rate}/"
class_list = [x.strip() for x in open('./{}/class.txt'.format(data_name), encoding='utf-8').readlines()]

class scoreAndWord(object):
    def __init__(self,score,word):
        self.score = score
        self.word = word

    def __lt__(self, other):  # 降序
        return self.score > other.score

def stopwordslist(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

def data_pro(name_list):
    # if data_name== "20News_bert" or data_name== "BBC_bert":
    if data_name in ["fudan", "qinghua"]:
        print("CHA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        stopwords = stopwordslist("./stopwords/cn_stopwords.txt")
    else:

        print("ENG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        stopwords = stopwordslist("./stopwords/en.txt")
    # if data_name in ["20News_bert", "Semeval_bert", "BBC_bert", "SST_bert"]:
    #     print("ENG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     stopwords = stopwordslist("./stopwords/en.txt")
    # else:
    #     print("CHA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    #     stopwords = stopwordslist("./stopwords/cn_stopwords.txt")

    for name in name_list:

        all_text = []
        all_label = []

        ############# fine-tune的train的修改
        # if name == "train":
        #     path = cur_path + '{}.json'.format(name)
        # else:
        #     path = fine_tune_path + '{}.json'.format(name)
        path = fine_tune_path + '{}.json'.format(name)

        with open(path, 'r', encoding='UTF-8') as f:
            all = json.load(f)
            print("正常是：", len(all))

            for data in all:
                #print(data)
                text = data['text']
                label = int(data['label'])

                sent_list = jieba.cut(text)
                out = []
                for word in sent_list:
                    if word not in stopwords:
                        if word != '\t' and word != '\n':
                            out.append(word)

                out = out[:500]
                out = " ".join(out)

                all_text.append(out)
                all_label.append(label)

        print(len(all_text))
        print(len(all_label))

        output = open(cur_path + "{}_contents.pkl".format(name), 'wb')
        pickle.dump(all_text, output)

        output = open(cur_path + "{}_labels.pkl".format(name), 'wb')
        pickle.dump(all_label, output)

def my_save(train_txt, train_labels, classify_num_labels):
    words_count = dict([(c, {}) for c in range(classify_num_labels)])
    total_word_count = {}

    all_N = len(train_txt)

    doc_count_by_category = dict([(c, 0) for c in range(classify_num_labels)])

    for i in range(len(train_txt)):

        text = train_txt[i].split(" ")
        label = int(train_labels[i])

        doc_count_by_category[label] += 1

        for word in text:
            if word == " " or word == "":
                continue
            words_count[label].setdefault(word, 0)
            words_count[label][word] += 1
            total_word_count.setdefault(word, 0)
            total_word_count[word] += 1

    output = open(cur_path + "words_count.pkl", 'wb')
    pickle.dump(words_count, output)

    output = open(cur_path + "total_word_count.pkl", 'wb')
    pickle.dump(total_word_count, output)

    output = open(cur_path + "doc_count_by_category.pkl", 'wb')
    pickle.dump(doc_count_by_category, output)

    # 计算每一类别下包含词的数量
    total_words_by_category = {}
    for category in range(classify_num_labels):
        total_words_by_category[category] = sum(words_count[category].values())

    output = open(cur_path + "total_words_by_category.pkl", 'wb')
    pickle.dump(total_words_by_category, output)

    # 计算每个词在每一类出现的文档数目
    word_doc_count = dict([(word, [0] * classify_num_labels) for word in total_word_count.keys()])
    for word in word_doc_count.keys():
        for i in range(len(train_txt)):
            text = train_txt[i]
            label = int(train_labels[i])

            if word in text:
                word_doc_count[word][label] += 1

    output = open(cur_path + "word_doc_count.pkl", 'wb')
    pickle.dump(word_doc_count, output)

def get_feature(data, word_list):
    """

    :param data:
    :param word_list: 2维，考虑类别
    :return:
    """
    res = []
    for doc in data:
        cur = []
        for i in range(len(word_list)):
            for j in range(len(word_list[0])):
                if word_list[i][j] in doc:
                    cur.append(1)
                else:
                    cur.append(0)

        res.append(deepcopy(cur))
    return res

def get_feature1(data, word_list):
    """
    word_list 一维，即不考虑类别信息
    :param data:
    :param word_list:
    :return:
    """
    res = []
    for doc in data:
        cur = []

        for i in range(len(word_list)):
            word = word_list[i]
            if word in doc:
                cur.append(1)
            else:
                cur.append(0)
        res.append(deepcopy(cur))
    return res

def get_feature3(data, word_list):
    res = []
    for doc in data:
        cur = []
        for i in range(len(word_list)):
            count = 0
            for j in range(len(word_list[i])):
                if word_list[i][j] in doc:
                    count += 1
            cur.append(count)

        res.append(deepcopy(cur))
    return res

def chi_new(train_txt, train_labels, test_txt, test_labels, val_txt, val_labels, classify_num_labels,
            words_count, total_word_count, total_words_by_category, word_doc_count, doc_count_by_category):
    """
    改进，不考虑类别
    :param train_txt:
    :param train_labels:
    :param test_txt:
    :param test_labels:
    :param classify_num_labels:
    :param words_count:
    :param total_word_count:
    :param total_words_by_category:
    :param word_doc_count:
    :param doc_count_by_category:
    :return:
    """
    all_N = len(train_txt)

    pq = PriorityQueue()

    for word in total_word_count.keys():
        for category in range(classify_num_labels):
            # A表示分类结果是c的文档中，包含词语t的文档数目
            a = word_doc_count[word][category]
            # B表示数据集中分类结果不是c的文档中，包含词语t的文档数目
            b = sum(word_doc_count[word]) - a
            # C表示分类结果是c的文档中，不包含词语t的文档数目
            c = doc_count_by_category[category] - a
            # D表示数据集中分类结果不是c的文档中，不包含词语t的文档数目
            d = sum(doc_count_by_category.values()) - doc_count_by_category[category] - b
            n = a + b + c + d

            chi_ori = 1.0 * n * (a * d - b * c) ** 2 / ((a + c) * (a + b) * (b + d) * (c + d) + 0.0001)

            # N11:表某一类别下包含某个词的数量
            N11 = words_count[category].get(word, 0)
            # # N10:表示其他类别包含某个词的数量
            N10 = total_word_count[word] - N11

            tf_new = float(N11) / (N10 + 1.0)
            tf_new = math.pow(tf_new, 1 / 3)

            # tf_new = math.pow(float(N11)  / (total_words_by_category[category]), 1/4)
            # tf_new = math.pow(float(N11), 1 / 10)

            b = (word_doc_count[word][category] / float(doc_count_by_category[category] + 0.0001))  # bata

            class_contain_t = 0  # 包含单词t的类别数目
            for i in word_doc_count[word]:
                if i > 0:
                    class_contain_t += 1
            e = math.tanh(classify_num_labels / float(class_contain_t))

            # df = sum(word_doc_count[word]) - word_doc_count[word][category]
            # idf = math.log(1 + word_doc_count[word][category] / float(df + 1.0))

            tf = float(N11) / (total_words_by_category[category] + 0.0001)

            df = sum(word_doc_count[word]) - word_doc_count[word][category]
            idf = math.log(1 + word_doc_count[word][category] / float(df + 1.0))

            tfidf = tf * idf * b * e

            chi = chi_ori * b * tf

            final = p * tfidf + (1-p) * chi

            # final = chi_ori * b * tf

            pq.put_nowait(scoreAndWord(final, word))

    word_list_chi = []
    count = 0
    while not pq.empty():
        cur = pq.get()
        if cur.word not in word_list_chi:
            word_list_chi.append(cur.word)
            count += 1
        if count == total_feature:
            break

    train_data_chi = get_feature1(train_txt, word_list_chi)
    test_data_chi = get_feature1(test_txt, word_list_chi)
    val_data_chi = get_feature1(val_txt, word_list_chi)

    # for i in range(len(word_list_chi)):
    #     print(word_list_chi[i], end = " ")
    # print()

    # count_all = []
    #
    # for data_set in ["train", "val", "test"]:
    #     paths = (
    #         pd.read_csv(join(cur_path, "ids_{}.csv".format(data_set)), header=None)
    #             .to_numpy()
    #             .flatten()
    #     )
    #
    #     labels = (
    #         pd.read_csv(
    #             join(cur_path, "targets_{}.csv".format(data_set)), header=None
    #         )
    #             .to_numpy()
    #             .flatten()
    #     )
    #
    #     for i in range(len(paths)):
    #         if data_set == "train":
    #             temp = deepcopy(train_data_chi[i])
    #             temp.insert(0, paths[i])
    #             temp.insert(1, labels[i])
    #
    #             count_all.append(temp)
    #
    #         if data_set == "test":
    #             temp = deepcopy(test_data_chi[i])
    #             temp.insert(0, paths[i])
    #             temp.insert(1, labels[i])
    #
    #             count_all.append(temp)
    #
    #         if data_set == "val":
    #             temp = deepcopy(val_data_chi[i])
    #             temp.insert(0, paths[i])
    #             temp.insert(1, labels[i])
    #
    #             count_all.append(temp)
    #
    # # Create the header of csv
    # header = ["data", "label"]
    # for i in range(len(word_list_chi)):
    #     print(word_list_chi[i])
    #     header.append("chi_token_{}".format(word_list_chi[i]))
    #
    # # Save the final csv
    # count_all.insert(0, header)
    # pd.DataFrame(count_all).to_csv(join(cur_path, "chi_token_new.csv"), header=None, index=None)




    # 训练模型
    svclf_chi = SVC(kernel='linear')  # 初始化SVM支持向量机模型
    svclf_chi.fit(train_data_chi, train_labels)  # 对训练集进行训练
    svm_predict_chi = svclf_chi.predict(test_data_chi)  # 对测试集进行预测

    labels_all = []
    predict_all = []
    for i in range(len(test_labels)):
        labels_all.append(int(test_labels[i]))
        predict_all.append(int(svm_predict_chi[i]))

    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    print("改进，不考虑类别的实验效果~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("chi_ori ACC = ", acc)
    print(report)
    print(confusion)
    print()

def tfidf_new(train_txt, train_labels, test_txt, test_labels, val_txt, val_labels, classify_num_labels,
            words_count, total_word_count, total_words_by_category, word_doc_count, doc_count_by_category):
    """
    固定特征数目，无关类别
    :param train_txt:
    :param train_labels:
    :param test_txt:
    :param test_labels:
    :param classify_num_labels:
    :param words_count:
    :param total_word_count:
    :param total_words_by_category:
    :param word_doc_count:
    :param doc_count_by_category:
    :return:
    """

    pq = PriorityQueue()

    all_N = len(train_txt)

    for word in total_word_count.keys():
        for category in range(classify_num_labels):
            # N11:表某一类别下包含某个词的数量
            N11 = words_count[category].get(word, 0)
            tf = float(N11) / (total_words_by_category[category])
            b = word_doc_count[word][category] / float(doc_count_by_category[category])  # bata


            class_contain_t = 0  # 包含单词t的类别数目
            for i in word_doc_count[word]:
                if i > 0:
                    class_contain_t += 1
            e = math.tanh(classify_num_labels / (class_contain_t))

            df = sum(word_doc_count[word]) - word_doc_count[word][category]
            idf = math.log(1.0 + word_doc_count[word][category] / float(df + 1.0))

            # df = sum(word_doc_count[word])  # 传统的计算方法
            # idf = math.log(all_N / float(df + 1.0))

            final = tf * idf * b * e  # 传统的计算方法

            pq.put_nowait(scoreAndWord(final, word))

    word_list_tfidf = []
    count = 0
    while not pq.empty():
        cur = pq.get()
        if cur.word not in word_list_tfidf:
            word_list_tfidf.append(cur.word)
            count += 1
        if count == total_feature:
            break

    train_data_tfidf = get_feature1(train_txt, word_list_tfidf)
    test_data_tfidf = get_feature1(test_txt, word_list_tfidf)

    for i in range(len(word_list_tfidf)):
        print(word_list_tfidf[i], end = " ")
    print()

    # # 1、实例化PCA, 整数——指定降维到的维数
    # transfer2 = PCA(n_components=100)
    # # 2、调用fit_transform
    # train_data_tfidf = transfer2.fit_transform(train_data_tfidf)
    # test_data_tfidf  =transfer2.transform(test_data_tfidf)

    # 训练模型
    svclf_tfidf = SVC(kernel='linear')  # 初始化SVM支持向量机模型
    svclf_tfidf.fit(train_data_tfidf, train_labels)  # 对训练集进行训练
    svm_predict_tfidf = svclf_tfidf.predict(test_data_tfidf)  # 对测试集进行预测

    labels_all = []
    predict_all = []
    for i in range(len(test_labels)):
        labels_all.append(int(test_labels[i]))
        predict_all.append(int(svm_predict_tfidf[i]))

    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)

    print("改进，不考虑类别的实验效果~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("tfidf_ori ACC = ", acc)
    print(report)
    print(confusion)
    print()

def chi_new_cate(train_txt, train_labels, test_txt, test_labels, val_txt, val_labels, classify_num_labels,
            words_count, total_word_count, total_words_by_category, word_doc_count, doc_count_by_category):
    """
    根据类别确定特征数目
    :param train_txt:
    :param train_labels:
    :param test_txt:
    :param test_labels:
    :param classify_num_labels:
    :param words_count:
    :param total_word_count:
    :param total_words_by_category:
    :param word_doc_count:
    :param doc_count_by_category:
    :return:
    """
    all_N = len(train_txt)

    chi_sq_by_category = dict([(c, {}) for c in range(classify_num_labels)])

    for word in total_word_count.keys():
        for category in range(classify_num_labels):
            # A表示分类结果是c的文档中，包含词语t的文档数目
            a = word_doc_count[word][category]
            # B表示数据集中分类结果不是c的文档中，包含词语t的文档数目
            b = sum(word_doc_count[word]) - a
            # C表示分类结果是c的文档中，不包含词语t的文档数目
            c = doc_count_by_category[category] - a
            # D表示数据集中分类结果不是c的文档中，不包含词语t的文档数目
            d = sum(doc_count_by_category.values()) - doc_count_by_category[category] - b
            n = a + b + c + d

            chi_ori = 1.0 * n * (a * d - b * c) ** 2 / ((a + c) * (a + b) * (b + d) * (c + d) + 1)

            # N11:表某一类别下包含某个词的数量
            N11 = words_count[category].get(word, 0)
            # # N10:表示其他类别包含某个词的数量
            N10 = total_word_count[word] - N11

            tf_new = float(N11) / (N10 + 1.0)
            tf_new = math.pow(tf_new, 1 / 3)

            # tf_new = math.pow(float(N11)  / (total_words_by_category[category]), 1/4)
            # tf_new = math.pow(float(N11), 1 / 10)

            b = (word_doc_count[word][category] / float(doc_count_by_category[category] + 1))  # bata

            class_contain_t = 0  # 包含单词t的类别数目
            for i in word_doc_count[word]:
                if i > 0:
                    class_contain_t += 1
            e = math.tanh(classify_num_labels / float(class_contain_t))

            # df = sum(word_doc_count[word]) - word_doc_count[word][category]
            # idf = math.log(1 + word_doc_count[word][category] / float(df + 1.0))

            tf = float(N11) / (total_words_by_category[category] + 1)

            final = chi_ori * b   * tf

            chi_sq_by_category[category][word] = final


    word_list_chi = []

    for category in range(classify_num_labels):
        word_chi_list = list(reversed(sorted(chi_sq_by_category[category].items(), key=lambda x: x[1])))[
                        :feature_per_class]
        cur = []
        for word, chi in word_chi_list:
            cur.append(word)

        word_list_chi.append(deepcopy(cur))


    train_data_chi = get_feature3(train_txt, word_list_chi)
    test_data_chi = get_feature3(test_txt, word_list_chi)
    val_data_chi = get_feature3(val_txt, word_list_chi)

    count_all = []

    for data_set in ["train", "val", "test"]:
        # if data_set=="train":
        #     path = cur_path
        # else:
        #     path = fine_tune_path
        path = fine_tune_path

        paths = (
            pd.read_csv(join(path, "ids_{}.csv".format(data_set)), header=None)
                .to_numpy()
                .flatten()
        )

        labels = (
            pd.read_csv(
                join(path, "targets_{}.csv".format(data_set)), header=None
            )
                .to_numpy()
                .flatten()
        )

        for i in range(len(paths)):
            if data_set == "train":
                temp = deepcopy(train_data_chi[i])

            if data_set == "test":
                temp = deepcopy(test_data_chi[i])

            if data_set == "val":
                temp = deepcopy(val_data_chi[i])

            # from sklearn import preprocessing
            # temp = preprocessing.normalize(np.array(temp).reshape(1, -1), norm='l2')
            # temp = temp.flatten()
            # temp = temp.tolist()

            temp.insert(0, paths[i])
            temp.insert(1, labels[i])

            count_all.append(temp)

    # Create the header of csv
    header = ["data", "label"]
    for i in range((classify_num_labels)):
        # print(word_list_chi[i])
        header.append("chi_token_class_{:0>3d}_count_token".format(i))

    # Save the final csv
    count_all.insert(0, header)
    pd.DataFrame(count_all).to_csv(join(cur_path, "chi_token_new.csv"), header=None, index=None)

    labels_all = []
    predict_all = []
    for i in range(1,len(count_all)):
        label = int(count_all[i][1])

        temp =  np.array(count_all[i][2:])
        index = np.argmax(temp)

        labels_all.append(label)
        predict_all.append(index)

    print("简单按照命中最多的分类：")
    acc = metrics.accuracy_score(labels_all, predict_all)
    report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    print("chi_ori ACC = ", acc)
    print(report)
    print(confusion)
    print()

    # # 训练模型
    # svclf_chi = SVC(kernel='linear')  # 初始化SVM支持向量机模型
    # svclf_chi.fit(train_data_chi, train_labels)  # 对训练集进行训练
    # svm_predict_chi = svclf_chi.predict(test_data_chi)  # 对测试集进行预测
    #
    # labels_all = []
    # predict_all = []
    # for i in range(len(test_labels)):
    #     labels_all.append(int(test_labels[i]))
    #     predict_all.append(int(svm_predict_chi[i]))
    #
    # acc = metrics.accuracy_score(labels_all, predict_all)
    # report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
    # confusion = metrics.confusion_matrix(labels_all, predict_all)
    #
    # print("改进，考虑类别的实验效果~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print("chi_ori ACC = ", acc)
    # print(report)
    # print(confusion)
    # print()

if __name__ == '__main__':

    data_pro(["train", "val", "test"])  # ["train", "val", "test"]
    print("data_pro over!!!")

    train_txt = pickle.load(open(cur_path + 'train_contents.pkl', 'rb'))
    test_txt = pickle.load(open(cur_path + 'test_contents.pkl', 'rb'))
    val_txt = pickle.load(open(cur_path + 'val_contents.pkl', 'rb'))

    train_labels = pickle.load(open(cur_path + 'train_labels.pkl', 'rb'))
    test_labels = pickle.load(open(cur_path + 'test_labels.pkl', 'rb'))
    val_labels = pickle.load(open(cur_path + 'val_labels.pkl', 'rb'))
    #
    my_save(train_txt, train_labels, classify_num_labels)
    print("my_save over!!!")

    words_count = pickle.load(open(cur_path + 'words_count.pkl', 'rb'))
    total_word_count = pickle.load(open(cur_path + 'total_word_count.pkl', 'rb'))
    total_words_by_category = pickle.load(open(cur_path + 'total_words_by_category.pkl', 'rb'))
    word_doc_count = pickle.load(open(cur_path + 'word_doc_count.pkl', 'rb'))
    doc_count_by_category = pickle.load(open(cur_path + 'doc_count_by_category.pkl', 'rb'))

    # tfidf_new(train_txt, train_labels, test_txt, test_labels, val_txt, val_labels, classify_num_labels,
    #           words_count, total_word_count, total_words_by_category, word_doc_count, doc_count_by_category)

    chi_new(train_txt, train_labels, test_txt, test_labels, val_txt, val_labels, classify_num_labels,
            words_count, total_word_count, total_words_by_category, word_doc_count, doc_count_by_category)

    chi_new_cate(train_txt, train_labels, test_txt, test_labels, val_txt, val_labels, classify_num_labels,
            words_count, total_word_count, total_words_by_category, word_doc_count, doc_count_by_category)

