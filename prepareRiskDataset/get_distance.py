
import os
from os.path import join

import numpy as np
from time import time
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm, trange


def shorten_paths(paths):
    for i in range(len(paths)):
        # print(paths[i])
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the distance to each class center of every data
def get_distance(
        layer, elem_name, num_class, csv_dir, bert_dir,fine_tune_dir, metric, data_sets=["train", "val", "test"]
):
    start = time()

    coordinate_train = pd.read_csv(
        os.path.join(bert_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
        header=None,
    ).to_numpy()
    label = pd.read_csv(
        os.path.join(bert_dir, "targets_{}.csv".format(data_sets[0])), header=None
    ).to_numpy().flatten()
    # coordinate_train = pd.read_csv(
    #     os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])),
    #     header=None,
    # ).to_numpy()
    # label = pd.read_csv(
    #     os.path.join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
    # ).to_numpy().flatten()

    # print(coordinate_train)
    # print(label)

    nc = NearestCentroid(metric=metric).fit(coordinate_train, label)
    centers = nc.centroids_

    for data_set in data_sets:
        # if data_set=="train":
        #     path = csv_dir
        # else:
        #     path = fine_tune_dir

        path = fine_tune_dir
        # Read coordinate and label of data
        # print(f"path:{path};layer:{layer}")
        # print(os.path.join(path, "distribution_{}_{}.csv".format(layer, data_set)))
        # print(os.path.join(csv_dir, "distribution_{}_{}.csv".format(layer, data_sets[0])))
        coordinates = pd.read_csv(
            os.path.join(path, "distribution_{}_{}.csv".format(layer, data_set)),
            header=None,
        ).to_numpy()
        labels = (
            pd.read_csv(
                os.path.join(path, "targets_{}.csv".format(data_set)), header=None
            )
                .to_numpy()
                .flatten()
        )
        paths = (
            pd.read_csv(
                os.path.join(path, "ids_{}.csv".format(data_set)), header=None
            )
                .to_numpy()
                .flatten()
        )
        shorten_paths(paths)

        # Calculate the distance to each class center of every data
        # print(f"dataset:{data_set}")
        # print(f"coordinates:{len(coordinates)}; centers:{len(centers)}")

        # print(len(coordinates))
        # print(len(centers))
        distance_to_center = pairwise.pairwise_distances(
            coordinates, centers, metric=metric
        ).tolist()
        #print(f"distance_center:{len(distance_to_center)}")

        # Insert path and label of data to csv
        for i in range(len(distance_to_center)):
            distance_to_center[i].insert(0, paths[i])
            distance_to_center[i].insert(1, labels[i])

        exec("distance_to_center_{} = distance_to_center".format(data_set))

    # Merge 3 csvs together
    distance_center_all = []

    for data_set in data_sets:
        exec("distance_center_all.extend(distance_to_center_{})".format(data_set))

    # Create the header of csv
    header = ["data", "label"]

    for i in range(num_class):
        header.append("{}_{}_class_{:0>3d}_distance".format(elem_name, layer, i))

    # Save the final csv
    distance_center_all.insert(0, header)
    pd.DataFrame(distance_center_all).to_csv(
        os.path.join(csv_dir, "{}_{}_distance.csv".format(elem_name, layer)),
        header=None, index=None
    )

    print("--- distance {:.2f} s ---".format(time() - start))


if __name__ == "__main__":
    get_distance()