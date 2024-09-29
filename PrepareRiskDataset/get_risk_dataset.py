import os
import shutil
from os.path import join
# import numpy as np
# import pandas as pd
# from PrepareRiskDataset.get_one_risk_info import get_one_risk_info
from PrepareRiskDataset.get_risk_info import get_risk_info
from PrepareRiskDataset.config.overall_config import *

data_dirs = [source_dataset]  # dataset dir
fine_tune_dirs = [target_dataset]
data_sets = ['train', 'val', "test"]
archive_dir = "."
elem_name_str = "{}"
# rate
note = str(rate)  # additional note to add in save folder, e.g., save 'fgvc_100_test' at archive_dir

csv_dir_str = "{}_{}/"+note # archive_dir +"/{}_{}/"+note  # where you save the distribution csv


def my_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# get risk elem
if __name__ == "__main__":

    for i in range(len(data_dirs)):
        data_dir = data_dirs[i]
        fine_tune_dir = fine_tune_dirs[i]
        print('\n===== {} ====='.format(data_dir))
        my_mkdir(join(archive_dir, data_dir))
        my_mkdir(join(archive_dir, data_dir, "risk_dataset{}".format(note)))
        my_mkdir(join(archive_dir, data_dir, "softmax"))
        my_mkdir(join(archive_dir, data_dir, "DBLP-Scholar"))
        my_mkdir(join(archive_dir, data_dir, "DBLP-Scholar", '325'))

        my_mkdir(join(archive_dir, '{}_cnn'.format(data_dir)))
        my_mkdir(join(archive_dir, '{}_cur'.format(data_dir)))

        get_risk_info(
            data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, note, token_name, fine_tune_dir
        )
        # get_one_risk_info(
        #     data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list
        # )

        # copy softmax to final save dir
        path = '{}/{}/{}/{}'.format(archive_dir, data_dir, 'softmax', note)
        bert_path = '{}_{}/{}'.format(fine_tune_dir, 'bert', note)
        # bert_path = '{}_{}/{}'.format(data_dir, 'bert', note)
        if not os.path.exists(path):
            os.makedirs(path)
        for cnn in cnns:
            my_mkdir(join(archive_dir, data_dir, "softmax", "{}".format(cnn)))
            for data_set in data_sets:
                shutil.copy(
                    join(
                        bert_path,
                        "distribution_dense_{}.csv".format(data_set),
                    ),
                    join(
                        path,
                        "distribution_dense_{}.csv".format(data_set),
                    ),
                )


        # # rule！
        # get_one_risk_info(data_dir, data_sets, cnns, layers, elem_name_str, elems, csv_dir_str, k_list, token_name)