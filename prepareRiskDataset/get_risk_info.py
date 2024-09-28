
import os
import shutil
from os.path import join

import numpy as np
import pandas as pd
from get_distance import get_distance
from get_knn_count import get_knn_count
from get_token_count import get_token_count
from get_cluster_count import get_cluster_count
from copy import deepcopy


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, note, token_name, fine_tune_dir
):
    # Get risk elem
    num_class_zh = 0

    # get each single risk element and save in {dataset}_{model}
    for layer in layers:

        for cnn in cnns:
            print("=== geting risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn) # 最终存储路径
            fine_tune_path = csv_dir_str.format(fine_tune_dir, cnn)
            bert_dir = fine_tune_path  # 选取的knn和distance中心数据的路径
            # bert_dir = csv_dir # 选取的knn和distance中心数据的路径
            # bert_dir = csv_dir_str.format(data_dir, 'bert')

            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)

            # num_class = int(np.max(pd.read_csv(join(bert_dir, "targets_{}.csv".format(data_sets[0])), header=None).to_numpy())) + 1
            num_class = int (np.max(pd.read_csv(join(fine_tune_path, "targets_{}.csv".format(data_sets[0])), header=None).to_numpy())) + 1

            num_class_zh = num_class

            # num_class = (
            #     int(
            #         pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            #         .to_numpy()
            #         .flatten()[-1]
            #     )
            #     + 1
            # )

            #get_distance(layer, elem_name, num_class, csv_dir, "cosine", data_sets)
            # print(f"csv:{csv_dir},fine_tune:{fine_tune_path}")
            get_distance(layer, elem_name, num_class, csv_dir, bert_dir, fine_tune_path, "cosine", data_sets)
            get_knn_count(
                k_list, layer, elem_name, num_class, csv_dir, bert_dir, fine_tune_path, "cosine", data_sets
            )

    #################################
    csv_dir = csv_dir_str.format(data_dir, "cur")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    # for tn in token_name:
    #     get_token_count(tn, num_class_zh, csv_dir)  # token

    # get_cluster_count(num_class_zh, csv_dir)  # count

    # Merge csv
    # saved in train datasets
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                csv_path_list.append(
                    join(
                        csv_dir_str.format(data_dir, cnn),
                        "{}_{}_{}.csv".format(cnn, layer, elem), # saved single risk element
                    )
                )

    #cnn = "cur"   # token
    for tn in token_name:
      csv_path_list.append(join(csv_dir_str.format(data_dir, 'bert'), "{}_token_new.csv".format(tn)))  # token
      #csv_path_list.append(join(csv_dir_str.format(data_dir, 'cur'), "{}_token_new.csv".format(tn)))  # token

    # csv_path_list.append(join(csv_dir_str.format(data_dir, cnn), "cluster_count.csv"))  # count

    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]

    for csv_path in csv_path_list:
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, 2:]
        print(csv_path)
        all_info = np.hstack((all_info, csv))

    pd.DataFrame(all_info).to_csv(
        "{}/all_data_info.csv".format(
            csv_dir
        ),
        header=None,
        index=None,
    )
    # pd.DataFrame(all_info).to_csv(
    #     "./{}/risk_dataset{}/pair_info_more.csv".format(
    #         data_dir, note
    #     ),
    #     header=None,
    #     index=None,
    # )

    all_info = all_info.tolist() # data_nums X feature_nums, feature_nums = feature1+feature2+...

    for data_set in data_sets:
        temp_csv = [all_info[0]]

        for line in all_info[1:]:

            if line[0].split("/")[0] == data_set:
                temp_csv.append(line)

        pd.DataFrame(deepcopy(temp_csv)).to_csv(
            "{}/{}.csv".format(
                csv_dir, data_set
            ),
            header=None,
            index=None,
        )


# get risk elem
if __name__ == "__main__":

    get_risk_info()