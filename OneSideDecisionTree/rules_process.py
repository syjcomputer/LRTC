import os
import logging
import operator
import numpy as np
import time
from multiprocessing import Pool


class Rule:
    def __init__(self, rule_id, rule_description):
        self.id = rule_id
        self.original_description = rule_description
        self.readable_description = ''
        self.conditions = dict()
        self.attr_op_2_value = dict()
        self.involved_attributes = set()
        self.infer_class = None  # str, 'U': unmatch, 'M': match
        self.match_number = 0
        self.unmatch_number = 0
        self.impurity = .0
        self.ops = {">": operator.gt, "<=": operator.le}
        self.__analysis_rule_text__()

    def __analysis_rule_text__(self):
        if len(self.original_description) > 0:
            origin_conditions = self.original_description.split(' && ')
            readable_conditions = []
            attr_op_2_values = dict()
            for condition in origin_conditions:
                condition_list = []
                condition_des = condition.split(':')
                elems = None
                compare_op = ''
                if '<=' in condition_des[0]:
                    elems = condition_des[0].split('<=')
                    compare_op = '<='
                elif '>' in condition_des[0]:
                    elems = condition_des[0].split('>')
                    compare_op = '>'
                else:
                    logging.raiseExceptions('Unknown comparator! Only "<=" or ">" allowed.')
                # Attribute metric
                condition_list.append(elems[0])
                self.involved_attributes.add(elems[0])
                # Comparator
                condition_list.append(compare_op)
                # Threshold
                condition_list.append(float(elems[1]))
                # More information
                condition_list.append(condition_des[1])
                self.conditions[condition_des[0]] = condition_list
                readable_conditions.append(elems[0] + compare_op + str(round(float(elems[1]), 4)))
                attr_op = str(elems[0]) + compare_op
                values = attr_op_2_values.get(attr_op)
                if values is None:
                    values = []
                    attr_op_2_values[attr_op] = values
                values.append(float(elems[1]))
            readable_text = ' && '.join(readable_conditions)
            '''
            Following part of codes aim to: 
            1) a>0.4 && a>0.5 --> a>0.5
            2) a<=0.3 && a<=0.2 --> a<=0.2
            '''
            for k, v in attr_op_2_values.items():
                if len(v) >= 2:
                    if '>' in k:
                        tight_threshold = np.max(v)
                    else:
                        tight_threshold = np.min(v)
                    self.attr_op_2_value[k] = tight_threshold
                    for value in v:
                        if value == tight_threshold:
                            continue
                        self.conditions.pop(k + str(value))
                        readable_cond = k + str(round(value, 4))
                        if readable_cond + ' && ' in readable_text:
                            readable_text = readable_text.replace(readable_cond + ' && ', '')
                        elif ' && ' + readable_cond in readable_text:
                            readable_text = readable_text.replace(' && ' + readable_cond, '')
                else:
                    self.attr_op_2_value[k] = v[0]
            last_condition = origin_conditions[len(origin_conditions) - 1]
            rule_info = last_condition.split(':')[1].split('|')
            readable_text += ' : ' + last_condition.split(':')[1]
            self.readable_description = readable_text
            self.infer_class = rule_info[0]
            self.id = self.infer_class + '_' + self.id
            self.unmatch_number = float(rule_info[1])
            self.match_number = float(rule_info[2])
            if len(rule_info) > 3:
                self.impurity = float(rule_info[3])
        else:
            logging.warning('Null rule is provided!')

    def __eq__(self, other):
        if not isinstance(other, Rule):
            logging.raiseExceptions('The input compare data is not Rule type!')
        conditions1 = sorted(self.conditions.keys())
        conditions2 = sorted(other.conditions.keys())
        if self.infer_class != other.infer_class:
            return False
        if len(conditions1) != len(conditions2):
            return False
        condition_str1 = ''.join(conditions1)
        condition_str2 = ''.join(conditions2)
        if condition_str1 == condition_str2:
            return True
        else:
            return False

    def __gt__(self, other):
        """
        A rule r_1 is greater than the other one r_2 iff
            1) their conclusions are the same, i.e., both match or unmatch;
            2) the conditions of r_1 are the subset of conditions of r_2. It usually means that the number of
            satisfied instances of r_1 will larger than that of r_2, so r_1 is greater than r_2.
        :param other:
        :return:
        """
        if not isinstance(other, Rule):
            logging.raiseExceptions('The input compare data is not Rule type!')
        conditions1 = sorted(self.conditions.keys())
        conditions2 = sorted(other.conditions.keys())
        if self.infer_class != other.infer_class:
            return False
        condition_str1 = ''.join(conditions1)
        condition_str2 = ''.join(conditions2)
        if condition_str1 == condition_str2:
            return False
        # simple string wyy-comparison.
        if condition_str1 in condition_str2:
            return True
        # complicate condition wyy-comparison: if conditions of r_1 dominate conditions of r_2, then r_1 > r_2.
        # 'a > 0.7' dominates 'a > 0.8'; 'a <= 0.4' dominates 'a <= 0.3'.
        for k1, v1 in self.attr_op_2_value.items():
            v2 = other.attr_op_2_value.get(k1)
            if v2 is None:
                return False
            else:
                if '>' in k1 and v1 > v2:
                    return False
                elif '<=' in k1 and v1 < v2:
                    return False
        return True

    def __lt__(self, other):
        if self == other:
            return False
        if self > other:
            return False
        return True

    def apply(self, attrs_2_values):
        """

        :param attrs_2_values: Type: dict(), attributes and the corresponding values.
        :return:
        """
        if attrs_2_values is None or len(attrs_2_values.keys()) != len(self.involved_attributes):
            logging.warning("The number of input attributes are not equal to {}'s requirement!".format(self.id))
            return 0
        for condition in self.conditions.values():
            attr_value = attrs_2_values[condition[0]]
            if attr_value is None:
                logging.warning("The value of attribute {} is None!".format(condition[0]))
                return 0
            if not self.ops[condition[1]](attr_value, condition[2]):
                return 0
        return 1


def save_rules(file_path, rules_list):
    """

    :param file_path:
    :param rules_list: [[metric, comparator, threshold, addition_info, metric, ...], [metric, ...], ...]
            metric: evaluation on attributes
            comparator: '<=', '>'
            threshold: real value between 0 and 1
            addition_info: U or M|unmatch_number|match_number|impurity
            toy example: [[title_jaccard_similarity, >=, 0.9, M|200|500|0.01]]
    :return:
    """
    if file_path is None:
        logging.raiseExceptions('Please set a file path!')
    if rules_list is None or len(rules_list) == 0:
        logging.raiseExceptions('No rules are provided!')
    file_ob = open(file_path, 'w')
    for rule in rules_list:
        file_ob.write(rule + '\n')
    file_ob.flush()
    file_ob.close()


def read_rules(file_path):
    """

    :param file_path:
    :return: A list of rules. Each rule is Rule type.
    """
    if file_path is None or not os.path.exists(file_path):
        logging.raiseExceptions('No rule files are found!')
    file_ob = open(file_path, 'r')
    rules = list()
    i = 0
    existing_items = set()
    for line in file_ob.readlines():
        rule_description = line.strip('\n')
        if rule_description in existing_items:
            continue
        rule_id = 'rule_' + str(i)
        rule = Rule(rule_id, rule_description)
        rules.append(rule)
        existing_items.add(rule_description)
        i += 1
    return rules


def clean_rules(rules, print_info=False):
    """
    Remove redundant rules.
    :param rules:
    :param print_info:
    :return:
    """
    cleaned_rules = []
    if print_info:
        print("- removing exact same rules! (#={})".format(len(rules)))
    # deplicate_rules = dict()
    rule_size = len(rules)
    _interval = int(np.maximum(0.01 * rule_size, 1))
    _start_time = time.time()
    for i in range(len(rules)):
        inspect_rule = rules[i]
        existence_flag = False
        for selected_rule in cleaned_rules:
            if selected_rule == inspect_rule:
                existence_flag = True
                # deplicate_rules[selected_rule.readable_description].append(inspect_rule.readable_description)
                break
        if existence_flag is False:
            cleaned_rules.append(inspect_rule)
            # deplicate_rules[inspect_rule.readable_description] = []
        else:
            pass
        if print_info and i >= _interval and (i % _interval == 0 or i == rule_size):
            print("Progress of remove same rules: {:.0%}, {}s".format(i / rule_size,
                                                                      time.time() - _start_time))
            _start_time = time.time()
    # for k, v in deplicate_rules.items():
    #     print("\n---- {} ----".format(k))
    #     for elem in v:
    #         print(elem)
    if print_info:
        print("- Done! (#={})".format(len(cleaned_rules)))
        print("- removing duplicated rules! (#={})".format(len(cleaned_rules)))
    cleaned_rules = merge_rules(cleaned_rules, print_info)
    if print_info:
        print("- Done! (#={})".format(len(cleaned_rules)))
    return cleaned_rules


def merge_rules(rules, print_info=False):
    """
    If a rule (e.g., r1) is contained by the other one (e.g. r2), then remove the longer one, i.e., r2.
    :param rules: list of rules.
    :param print_info:
    :return:
    """
    cleaned_rules = []
    remove_rule_ids = set()
    length_sorted_rules = []
    for rule in rules:
        length_sorted_rules.append([rule, len(rule.conditions.keys())])
    length_sorted_rules = sorted(length_sorted_rules, key=lambda item: item[1])
    rule_size = len(length_sorted_rules)
    _interval = int(np.maximum(0.01 * rule_size, 1))
    _start_time = time.time()
    for i in range(rule_size-1):
        if length_sorted_rules[i] == -1:
            continue
        probe_rule = length_sorted_rules[i][0]
        for j in range(i+1, rule_size):
            if length_sorted_rules[j] == -1:
                continue
            target_rule = length_sorted_rules[j][0]
            if probe_rule > target_rule:  # for rule wyy-comparison, the shorter rule is larger than the longer one.
                remove_rule_ids.add(target_rule.original_description)
                length_sorted_rules[j] = -1
            elif target_rule > probe_rule:
                remove_rule_ids.add(probe_rule.original_description)
                length_sorted_rules[i] = -1
                probe_rule = target_rule
        if print_info and i >= _interval and (i % _interval == 0 or i == rule_size):
            print("Progress of de-duplicate rules: {:.0%}, {}s".format(i / rule_size,
                                                                       time.time() - _start_time))
            _start_time = time.time()
    for rule in rules:
        if rule.original_description not in remove_rule_ids:
            cleaned_rules.append(rule)
    return cleaned_rules


def clean_rules_mt(rules, process_number=5):
    """
    Remove redundant rules. Multiple processes version.
    :param rules:
    :param process_number:
    :return:
    """
    n = len(rules)
    p = process_number
    bs = int(n / p)
    if bs == 0:
        bs = 1
    batch_num = n // bs + (1 if n % bs else 0)
    # for i in range(batch_num):
    #     print((i * bs), min((i + 1) * bs, n))
    pool = Pool(p)
    _start_time = time.time()
    print("- start multiple rule cleaning processes: {}".format(p))
    print("- before mt clean rules! (#={})".format(n))
    return_rules = pool.map(clean_rules, [rules[(i * bs): min((i + 1) * bs, n)] for i in range(batch_num)])
    pool.close()
    pool.join()
    print("- Done! Elapsed Time: {}s.".format(time.time() - _start_time))
    rules_temp = []
    for i in range(len(return_rules)):
        rules_temp.extend(return_rules[i])
    print("- after mt clean rules! (#={})".format(len(rules_temp)))
    results = clean_rules(rules_temp, True)
    return results


def select_rules_based_on_threshold(rules, match_threshold, unmatch_threshold):
    """
    Those rules that their impurities are higher than the thresholds will be removed.
    :param rules:
    :param match_threshold:
    :param unmatch_threshold:
    :return:
    """
    selected_rules = []
    for rule in rules:
        if rule.infer_class == 'M' and rule.impurity > match_threshold:
            continue
        elif rule.infer_class == 'U' and rule.impurity > unmatch_threshold:
            continue
        else:
            selected_rules.append(rule)
    return selected_rules


def test():
    f_path = 'rule_io_test.csv'
    rules = ['attr1>0.8:M|200|800|0.01',
             'attr1>0.8:M|100|800|0.01',
             'attr1<=0.8:U|1000|300|0.02 && attr2>0.9:M|50|250|0.03',
             'attr2>0.9:U|50|250|0.03 && attr1<=0.8:M|1000|300|0.02',
             'attr1<=0.8:U|1000|300|0.02 && attr2>0.9:M|50|250|0.03 && attr3>0.85:M|45|255|0.028',
             'attr3>0.85:M|45|255|0.028',
             'attr4<=0.1:M|0|0|0.1 && attr3>0.85:M|45|255|0.028',
             'attr3>0.85:M|45|255|0.028 && attr5<=0.01:U|0|0|0.01',
             'attr6>0.4:M|0|0|0 && attr6>0.5:M|0|0|0',
             'attr6>0.5:M|0|0|0 && attr6>0.4:M|0|0|0',
             'attr7<=0.3:U|0|0|0 && attr7<=0.2:U|0|0|0']
    save_rules(f_path, rules)
    rules_ = read_rules(f_path)
    print('\n'.join([r.original_description for r in rules_]))
    clean_rules_ = clean_rules(rules_)
    print('\nCleaned rules:')
    print('\n'.join([r.readable_description for r in clean_rules_]))


# def test_rule_apply():
#     f_path = 'rule_io_test.csv'
#     rules = read_rules(f_path)
#     rules = clean_rules(rules)
#     for rule in rules:
#         print(rule.id + ': ' + ' && '.join([elem for elem in rule.conditions.keys()]))
#     test_data = [{'attr1': 0.9},
#                  {'attr1': 0.7},
#                  {'attr1': 0.65, 'attr2': 0.95},
#                  {'attr1': 0.5, 'attr2': 0.91}
#                  ]
#     print("Apply rules:")
#     k = 0
#     print('{} with {}: {}'.format(rules[k].id, test_data[0], rules[k].apply(test_data[0])))
#     print('{} with {}: {}'.format(rules[k].id, test_data[1], rules[k].apply(test_data[1])))
#     k = 1
#     print('{} with {}: {}'.format(rules[k].id, test_data[2], rules[k].apply(test_data[2])))
#     k = 2
#     print('{} with {}: {}'.format(rules[k].id, test_data[3], rules[k].apply(test_data[3])))


def test_dominate_clean():
    f_path = 'rule_io_test.csv'
    rules = ['attr1>9.8:M|9|9|9',
             'attr1>9.7:M|9|9|9',
             'attr1>9.6:M|9|9|9',
             'attr2<=9.3:M|9|9|9',
             'attr2<=9.5:M|9|9|9',
             'attr3>9.9:M|9|9|9 && attr1>9.8:M|9|9|9',
             'attr4<=9.2:M|9|9|9 && attr3>9.9:M|9|9|9 && attr1>9.8:U|9|9|9',
             'attr5>9.6:M|9|9|9 && attr6>9.6:M|9|9|9',
             'attr6>9.4:M|9|9|9']
    save_rules(f_path, rules)
    rules_ = read_rules(f_path)
    print('\n'.join([r.original_description for r in rules_]))
    clean_rules_ = clean_rules(rules_)
    print('\nCleaned rules:')
    print('\n'.join([r.readable_description for r in clean_rules_]))


if __name__ == '__main__':
    # test()
    # test_rule_apply()
    test_dominate_clean()
