
from os.path import join
import os
import numpy as np
import pandas as pd
from PrepareRiskDataset.get_one_distance import get_one_distance
from PrepareRiskDataset.get_one_knn_count import get_one_knn_count
from PrepareRiskDataset.get_one_token_count import get_one_token_count
from PrepareRiskDataset.get_one_cluster_count import get_one_cluster_count


def get_one_risk_info(
    data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, token_name
):
    num_class_zh = 0
    # get risk elem
    for layer in layers:
        for cnn in cnns:
            print("=== geting one_risk_info of {}_{} ===".format(cnn, layer))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(data_dir, cnn)

            num_class = int(
                np.max(pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None).to_numpy())) + 1

            num_class_zh = num_class

            # num_class = (
            #     int(
            #         pd.read_csv(join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None)
            #         .to_numpy()
            #         .flatten()[-1]
            #     )
            #     + 1
            # )

            get_one_distance(layer, elem_name, num_class, csv_dir, "cosine", data_sets)
            get_one_knn_count(
                k_list, layer, elem_name, num_class, csv_dir, "cosine", data_sets
            )

    csv_dir = csv_dir_str.format(data_dir, "cur")
    for tn in token_name:
        # get_token_count(tn, csv_dir, num_class_zh)  # token
        get_one_token_count(tn, num_class_zh, csv_dir)    # token

    # get_one_cluster_count(num_class_zh, csv_dir)

    # Merge csv
    csv_path_list = []
    for cnn in cnns:
        for layer in layers:
            for elem in elems:
                csv_path_list.append(
                    join(
                        csv_dir_str.format(data_dir, cnn),
                        "{}_{}_one_{}.csv".format(cnn, layer, elem),
                    )
                )

    cnn = "cur"  # token
    for tn in token_name:
        csv_path_list.append(join(csv_dir_str.format(data_dir, cnn), "{}_one_token.csv".format(tn)))  # token

    # csv_path_list.append(join(csv_dir_str.format(data_dir, cnn), "cluster_one_count.csv"))

    # cnn = "bert"    # token
    # csv_path_list.append(join(csv_dir_str.format(data_dir, cnn), "bert_one_token.csv"))    # token

    all_info = pd.read_csv(csv_path_list[0], header=None).to_numpy()[:, :2]

    for csv_path in csv_path_list:
        csv = pd.read_csv(csv_path, header=None).to_numpy()[:, -1:]
        all_info = np.hstack((all_info, csv))

    # pd.DataFrame(all_info).to_csv(
    #     "./{}/DBLP-Scholar/all_data_info.csv".format(data_dir),  #  pair_info_more
    #     # "./{}/DBLP-Scholar/pair_info_more.csv".format(data_dir),  # pair_info_more
    #     header=None,
    #     index=None,
    # )
    pd.DataFrame(all_info).to_csv(
        "./{}/DBLP-Scholar/pair_info_more.csv".format(data_dir),  # pair_info_more
        header=None,
        index=None,
    )

    all_info = all_info.tolist()

    # for data_set in ['train', 'val', 'test']:
    for data_set in ["train"]:
        temp_csv = [all_info[0]]

        for line in all_info[1:]:

            if line[0].split("/")[0] == data_set:
                temp_csv.append(line)

        pd.DataFrame(temp_csv).to_csv(
            "./{}/DBLP-Scholar/325/{}.csv".format(
                data_dir, data_set
            ),
            header=None,
            index=None,
        )

def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
if __name__ == "__main__":


    get_one_risk_info()