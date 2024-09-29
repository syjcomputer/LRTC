
import multiprocessing
import os
from functools import partial
from itertools import combinations
from os.path import join

import numpy as np
import pandas as pd
from time import time
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


def get_knn(k, list, label_train):
    list_sorted = np.sort(list)
    ans = np.empty(k, dtype = int)

    for i in range(k):
        idx = np.where(list == list_sorted[i])[0][0]
        ans[i] = label_train[idx]

    return ans


# Calculate the knn_count to each class center of every data
def get_knn_count(
    k_list,
    layer,
    elem_name,
    num_class,
    csv_dir,
    bert_dir,
    fine_tune_dir,
    metric,
    data_sets=["train", "val", "test"],
):

    for k in k_list:
        start = time()

        # Get knn_count for all data_sets
        count_all = []
        for data_set in data_sets:
            # if data_set=="train":
            #     path = csv_dir
            # else:
            #     path = fine_tune_dir

            path = fine_tune_dir
            path2 = fine_tune_dir
            labels_train = (
                pd.read_csv(
                    join(path2, "targets_{}.csv".format(data_sets[0])), header=None
                )
                .to_numpy()
                .flatten()
            )
            distribution_train = pd.read_csv(
                join(path2, "distribution_{}_{}.csv".format(layer, data_sets[0])),
                header=None,
            ).to_numpy()
            paths = (
                pd.read_csv(join(path, "ids_{}.csv".format(data_set)), header=None)
                .to_numpy()
                .flatten()
            )
            shorten_paths(paths)
            labels = (
                pd.read_csv(
                    join(path, "targets_{}.csv".format(data_set)), header=None
                )
                .to_numpy()
                .flatten()
            )

            distribution = pd.read_csv(
                join(path, "distribution_{}_{}.csv".format(layer, data_set)),
                header=None,
            ).to_numpy()

            # Get pairwise_distances for all
            # print(f"distribution:{len(distribution)}; distribution_train:{len(distribution_train)}")
            p_d = pairwise.pairwise_distances(distribution, distribution_train, metric=metric)

            # Get knn for all
            s_p_d = np.sort(p_d)
            knn = np.empty([len(p_d), k], dtype=int)
            for i in range(len(p_d)):
                for j in range(k):
                    knn[i, j] = labels_train[np.where(p_d[i] == s_p_d[i, j])[0][0]]

            # Get knn count for all
            count = [np.bincount(knn[i], minlength=num_class).tolist() for i in range(len(knn))]

            # Combine all knn_count
            for i in range(len(count)):
                count[i].insert(0, paths[i])
                count[i].insert(1, labels[i])

            count_all.extend(count)
            # print(f"{data_set}; K:{k};  elem:{elem_name};   layer:{layer}; count:{len(count)}")


        # Create the header of csv
        header = ["data", "label"]
        for i in range(num_class):
            header.append("{}_{}_class_{:0>3d}_count{}".format(elem_name, layer, i, k))

        # Save the final csv
        count_all.insert(0, header)
        pd.DataFrame(count_all).to_csv(
            os.path.join(csv_dir, "{}_{}_knn{}.csv".format(elem_name, layer, k)), # elem_name = cnn, "{}_{}_{}.csv".format(cnn, layer, elem),
            header=None, index=None
        )

        print("--- knn_{} {:.2f} s ---".format(k, time() - start))


if __name__ == "__main__":

    get_knn_count()