import os
import pandas as pd
import csv
import sys
import numpy as np

def rules_merge(input_path, output_path, class_num):
    rules_clean_path = os.path.join(output_path, 'decision_tree_rules_clean.txt')
    rules_info_path = os.path.join(output_path, 'decision_tree_rules_info.txt')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ucnt = 0
    mcnt = 0

    with open(rules_clean_path, 'w') as fw:
        for i in range(class_num):
            filename = str(i) + '_decision_tree_rules_clean.txt'
            path = os.path.join(input_path, filename)
            with open(path, 'r') as fr:
                lines = fr.readlines()
                for line in lines:
                    line = line.replace('M', 'M' + str(i))
                    line = line.replace('U', 'U' + str(i))
                    fw.write(line)


    #need update
    with open(rules_info_path, 'w') as fw:
        for i in range(class_num):
            filename = str(i) + '_decision_tree_rules_info.txt'
            path = os.path.join(input_path, filename)
            with open(path, 'r') as fr:
                lines = fr.readlines()
                for line in lines:
                    line = line.replace('M', 'M' + str(i))
                    line = line.replace('U', 'U' + str(i))
                    fw.write(line)


def equal_column(x, y):
    for i in range(1, len(x)):
        if x[i] != y[i]:
            return False

    return True


def delete_similar_columns(path):
    f = csv.reader(open(path, 'r'))
    data = []
    delete = []
    for line in f:
        data.append(line)
    data = np.array(data)
    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            x = data[:, i]
            y = data[:, j]
            ans = equal_column(x, y)
            if ans:
                delete.append(j)

    data = np.delete(data, delete, axis=1)
    with open(path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(data.tolist())


# delete_similar_columns('/home/ltw/Documents/research/datasets/bird-40/data2csv/den121/1_val.csv')