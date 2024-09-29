import numpy as np
import csv
import sys
import os

class write_2_risk_csv():
    def __init__(self, data_setting):
        self.input_paths = data_setting.input_paths
        self.output_paths = data_setting.output_paths
        self.dtype = data_setting.dtype
        self.class_num = data_setting.class_num
        self.base_risk_num = data_setting.base_risk_num

        self.train_file = np.load(data_setting.train_file_paths)
        self.train_all_sim = np.load(data_setting.train_all_sim)
        self.train_all_dis = np.load(data_setting.train_all_dis)
        self.train_labels = np.load(data_setting.train_labels)

        self.val_file = np.load(data_setting.val_file_paths)
        self.val_all_sim = np.load(data_setting.val_all_sim)
        self.val_all_dis = np.load(data_setting.val_all_dis)
        self.val_labels = np.load(data_setting.val_labels)

        self.test_file = np.load(data_setting.test_file_paths)
        self.test_all_sim = np.load(data_setting.test_all_sim)
        self.test_all_dis = np.load(data_setting.test_all_dis)
        self.test_labels = np.load(data_setting.test_labels)

        self.file = [self.train_file, self.val_file, self.test_file]
        self.labels = [self.train_labels, self.val_labels, self.test_labels]
        self.all_sim = [self.train_all_sim, self.val_all_sim, self.test_all_sim]
        self.all_dis = [self.train_all_dis, self.val_all_dis, self.test_all_dis]

        self.write_data = [data_setting.write_train, data_setting.write_val, data_setting.write_test]
        self.all_date = data_setting.write_all_data_info


        if not os.path.exists(self.output_paths):
            os.makedirs(self.output_paths)

    def write_csv(self, path, headers, rows):
        out_path = os.path.join(self.output_paths, path)
        with open(out_path, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(rows)


    def write_info(self, base_risk_num, types):  # types 0:train 1:val 2:test
        rows = []

        num = len(self.labels[types])

        for i in range(num):
            row = []
            row.append(self.file[types][i])
            row.append(self.labels[types][i])
            if self.dtype == 'sim':
                sim = self.all_sim[types][i]
                row.extend(sim)
            else:
                dis = self.all_dis[types][i]
                row.extend(dis)

            rows.append(row)

        return rows

    def write_infos(self, risk_base_num):
        headers = []
        headers.append('')
        headers.append('label')
        for i in range(self.class_num):
            for j in range(risk_base_num):
                dnn_name = self.input_paths.split('/')[-1]
                headers.append(dnn_name + '_class' + str(i) + 'proto' + str(j))

        rows = []
        for i in range(3):
            row = self.write_info(risk_base_num, i)
            self.write_csv(self.write_data[i], headers, row)
            rows.extend(row)

        self.write_csv(self.all_date, headers, rows)


    def write_all(self):
        self.write_infos(self.base_risk_num)


