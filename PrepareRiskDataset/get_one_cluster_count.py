
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

def get_one_cluster_count(
    num_class,
    csv_dir,
    data_sets=["train", "val", "test"]
):
    start = time()
    count_all = []

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
        cluster = pd.read_csv(
            join(csv_dir, "bert_cluster_{}.csv".format(data_set)),
            header=None,
        ).to_numpy()


        for i in range(len(paths)):
            data = cluster[i]
            count = [0] * num_class
            temp = []
            for j in range(len(data)):
                count[j] = data[j]

            # count.insert(0, paths[i])
            # count.insert(1, labels[i])

            for k in range(num_class):
                temp.append(
                    [
                        "{}_{:0>3d}".format(paths[i], k),
                        "{}".format(1 if k == labels[i] else 0),
                        count[k],
                    ]
                )

            count_all.extend(temp)


    header = ["data", "label", "cluster_count"]

    # Save the final csv
    count_all.insert(0, header)
    pd.DataFrame(count_all).to_csv(
        os.path.join(csv_dir, "cluster_one_count.csv"),
        header=None, index=None
    )

    print("--- cluster_one_count {:.2f} s ---".format(time() - start))


if __name__ == "__main__":
    get_one_cluster_count(20, "./20News_bert")