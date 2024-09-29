
import multiprocessing
import os
from functools import partial
from itertools import combinations
from os.path import join

import numpy as np
import pandas as pd
from time import time
import pickle
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange
def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])

def get_one_token_count(
    tn,
    num_class,
    csv_dir,
    data_sets=["train", "val", "test"]
):
    start = time()
    count_all = []
    all_data = pickle.load(open(csv_dir + '/all_data.pkl', 'rb'))
    # token = pickle.load(open(csv_dir + '/token.pkl', 'rb'))

    df_train_pre = pd.read_csv(csv_dir + '/{}_words.csv'.format(tn), header=None)
    train_pre = df_train_pre.values
    token_dic = dict()
    for i in train_pre:
        label = int(i[2])
        token_dic.setdefault(label, [])
        # if len(token_dic[label]) >= 21:
        #     continue
        token_dic[label].append(i[0])

    for data_set in data_sets:
        paths = (
            pd.read_csv(join(csv_dir, "ids_{}.csv".format(data_set)), header=None)
                .to_numpy()
                .flatten()
        )

        labels = (
            pd.read_csv(
                join(csv_dir, "targets_{}.csv".format(data_set)), header=None
            )
                .to_numpy()
                .flatten()
        )



        for i in range(len(paths)) :
            id = paths[i]
            text = all_data[id]
            count = [0] * num_class
            temp = []

            for k, v in token_dic.items():
                for word in v:
                    if word in text:
                        count[k] += 1

            # for j in range(len(token)):
            #     for to in token[j]:
            #         if to in text:
            #             count[j] += 1

            for k in range(num_class):
                temp.append(
                    [
                        "{}_{:0>3d}".format(paths[i], k),
                        "{}".format(1 if k == labels[i] else 0),
                        count[k],
                    ]
                )

            count_all.extend(temp)


    header = ["data", "label", "{}_token_count".format(tn)]

    # Save the final csv
    count_all.insert(0, header)
    pd.DataFrame(count_all).to_csv(
        os.path.join(csv_dir, "{}_one_token.csv".format(tn)),
        header=None, index=None
    )

    print("--- token {:.2f} s ---".format(time() - start))


if __name__ == "__main__":
    get_one_token_count("bert", 20, "./20News_bert")