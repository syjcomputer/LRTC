from tqdm import tqdm

from RiskAnalysis.common import config
import logging
import os

from RiskAnalysis.data_process.data_merge import datamerge
from RiskAnalysis.data_process.data_setting import data_setting
from RiskAnalysis.data_process.decision_tree_2_rules import generate_all_rules
from RiskAnalysis.data_process.rules_merge import rules_merge, delete_similar_columns
from RiskAnalysis.data_process.write2riskcsv import write_2_risk_csv
from RiskAnalysis.data_process.write2rulescsv import write_2_rules_csv

cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")
class_num = cfg.get_class_num()
base_risk_num = cfg.base_risk_nums

def generate_riskdata():
    """
    :return:
    generate riskdata from npydataset
    """
    if not cfg.generate_rules:
        pass
    else:
        npy_input = cfg.get_npy_dataset_path()
        csv_output = cfg.get_data2csv_path()
        mulcsv_output = cfg.get_data2mulcsv_path()
        rules_output = cfg.get_rules_dataset_path()
        risk_path = cfg.get_risk_dataset_path()

        base_risk_list = cfg.base_risk_list
        csv_output_path = []
        for base_risk in base_risk_list:
            csv_output_path.append(os.path.join(csv_output, base_risk))

        for base_risk in base_risk_list:
            input_path = os.path.join(npy_input, base_risk)
            csv_output_path = os.path.join(csv_output, base_risk)
            mul_csv_path = os.path.join(mulcsv_output, base_risk)
            ds = data_setting(input_path, csv_output_path, class_num, base_risk_num)
            write_2_rules_csv(ds).write_all(base_risk)

        # merge data for rules
        output = os.path.join(csv_output, 'all')
        if not os.path.exists(output):
            os.makedirs(os.path.join(csv_output, 'all'))

        for i in tqdm(range(class_num), desc="generate csv of each class"):
            inputs = []
            for base_risk in base_risk_list:
                path = os.path.join(csv_output, base_risk)
                inputs.append(os.path.join(path, str(i) + '_train.csv'))
            datamerge(os.path.join(output, str(i) + '_train.csv'), base_risk_num, inputs)
            delete_similar_columns(os.path.join(output, str(i) + '_train.csv'))

            inputs = []
            for base_risk in base_risk_list:
                path = os.path.join(csv_output, base_risk)
                inputs.append(os.path.join(path, str(i) + '_val.csv'))
            datamerge(os.path.join(output, str(i) + '_val.csv'), base_risk_num, inputs)
            delete_similar_columns(os.path.join(output, str(i) + '_val.csv'))

            inputs = []
            for base_risk in base_risk_list:
                path = os.path.join(csv_output, base_risk)
                inputs.append(os.path.join(path, str(i) + '_test.csv'))
            datamerge(os.path.join(output, str(i) + '_test.csv'), base_risk_num, inputs)
            delete_similar_columns(os.path.join(output, str(i) + '_test.csv'))

            inputs = []
            for base_risk in base_risk_list:
                path = os.path.join(csv_output, base_risk)
                inputs.append(os.path.join(path, str(i) + '_all_data_info.csv'))
            datamerge(os.path.join(output, str(i) + '_all_data_info.csv'), base_risk_num, inputs)
            delete_similar_columns(os.path.join(output, str(i) + '_all_data_info.csv'))

        # generate rule
        if not os.path.exists(rules_output):
            os.makedirs(rules_output)
        generate_all_rules(output, rules_output, class_num, cfg.match_gini, cfg.unmatch_gini, cfg.tree_depth)

        # merge rule
        rules_merge(rules_output, risk_path, class_num)

        # generate data for risk model
        for base_risk in base_risk_list:
            input_path = os.path.join(npy_input, base_risk)
            mul_csv_path = os.path.join(mulcsv_output, base_risk)
            ds = data_setting(input_path, mul_csv_path, class_num, base_risk_num)
            write_2_risk_csv(ds).write_all()

        # merge data for risk model
        logging.info('merge train')
        inputs = []
        for base_risk in base_risk_list:
            path = os.path.join(mulcsv_output, base_risk)
            inputs.append(os.path.join(path, 'train.csv'))
        datamerge(os.path.join(risk_path, 'train.csv'), class_num * base_risk_num, inputs)

        logging.info('merge val')
        inputs = []
        for base_risk in base_risk_list:
            path = os.path.join(mulcsv_output, base_risk)
            inputs.append(os.path.join(path, 'val.csv'))
        datamerge(os.path.join(risk_path, 'val.csv'), class_num * base_risk_num, inputs)

        logging.info('merge test')
        inputs = []
        for base_risk in base_risk_list:
            path = os.path.join(mulcsv_output, base_risk)
            inputs.append(os.path.join(path, 'test.csv'))
        datamerge(os.path.join(risk_path, 'test.csv'), class_num * base_risk_num, inputs)

        logging.info('merge all')
        inputs = []
        for base_risk in base_risk_list:
            path = os.path.join(mulcsv_output, base_risk)
            inputs.append(os.path.join(path, 'all_data_info.csv'))
        datamerge(os.path.join(risk_path, 'all_data_info.csv'), class_num * base_risk_num, inputs)

