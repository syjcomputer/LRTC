import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import pandas as pd
import datetime

from sklearn import tree
from collections import Counter
import OneSideDecisionTree.rules_process
from tqdm import tqdm
from OneSideDecisionTree.common import config
from OneSideDecisionTree import rules_process

from PrepareRiskDataset.config.overall_config import classify_num_labels,source_dataset

data_name = source_dataset
data_path = f'{rootPath}/PrepareRiskDataset/{source_dataset}_cur'
class_num = classify_num_labels

cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection, config.global_risk_dataset)

rate = '1.0'
max_tree_depth = 4
match_tree_split_threshold = 0.4
unmatch_tree_split_threshold = 0.4
set_class_weight = 1000

match_rule_threshold = 0.4
unmatch_rule_threshold = 0.4

macth_gini = 0.03  # max 0.5  20news
unmatch_gini = 0.3
max_depth = 3

class ratree:
    trees = None
    pairtable = None
    newX = None
    newY = None
    featurenames = None
    currentdepth = None
    featurenameindex = None
    continuesplitconditions = None
    thetrees = None
    recursivepath = None
    featuretrees = None
    featurerulelists = None
    featurerulelistdescriptions = None
    class_names = ['U', 'M']

    def createfeaturetrees(self):
        for eachtree in self.trees:
            featuretree = []
            for nodeindex in range(1, len(eachtree)):
                splittree = eachtree[nodeindex - 1]
                splitfeature = eachtree[nodeindex].splitmetafeature
                thesplittree = splittree.thetrees[splitfeature].tree_
                splitthreshold = thesplittree.threshold[0]
                splitcondition = splittree.continuesplitconditions[splitfeature]
                splitdecision = None
                continuepath = None
                class_value = None
                continue_value = None
                class_impurity = None
                continue_class_impurity = None
                if splitcondition['left'] == 'D':
                    splitdecision = '<='
                    if thesplittree.n_outputs == 1:
                        class_value = thesplittree.value[1][0, :]
                    else:
                        class_value = thesplittree.value[1]
                    class_impurity = thesplittree.impurity[1]
                if splitdecision is not None:
                    class_name = ratree.class_names[np.argmax(class_value)]
                    class_value = str(class_value[0]) + '|' + str(class_value[1]) + '|' + str(class_impurity)
                    featurerulelistdescription = ''
                    featurerulelist = list(featuretree)
                    featurerulelist.append([splitfeature, splitdecision, splitthreshold, class_name, class_value])
                    self.featurerulelists.append(featurerulelist)
                    for eachfeaturerule in featurerulelist:
                        if len(featurerulelistdescription) > 0:
                            featurerulelistdescription += ' && '
                        featurerulelistdescription += (str(eachfeaturerule[0]) + eachfeaturerule[1] +
                                                       str(eachfeaturerule[2]) + ':' + str(eachfeaturerule[3]) +
                                                       '|' + str(eachfeaturerule[4]))
                    self.featurerulelistdescriptions.append(featurerulelistdescription)

                splitdecision = None
                if splitcondition['right'] == 'D':
                    splitdecision = '>'
                    if thesplittree.n_outputs == 1:
                        class_value = thesplittree.value[2][0, :]
                    else:
                        class_value = thesplittree.value[2]
                    class_impurity = thesplittree.impurity[2]
                if splitdecision is not None:
                    class_name = ratree.class_names[np.argmax(class_value)]
                    class_value = str(class_value[0]) + '|' + str(class_value[1]) + '|' + str(class_impurity)
                    featurerulelistdescription = ''
                    featurerulelist = list(featuretree)
                    featurerulelist.append([splitfeature, splitdecision, splitthreshold, class_name, class_value])
                    self.featurerulelists.append(featurerulelist)
                    for eachfeaturerule in featurerulelist:
                        if len(featurerulelistdescription) > 0:
                            featurerulelistdescription += ' && '
                        featurerulelistdescription += (str(eachfeaturerule[0]) + eachfeaturerule[1] +
                                                       str(eachfeaturerule[2]) + ':' + str(eachfeaturerule[3]) +
                                                       '|' + str(eachfeaturerule[4]))
                    self.featurerulelistdescriptions.append(featurerulelistdescription)

                if splitcondition['left'] == 'continue':
                    continuepath = '<='
                    if thesplittree.n_outputs == 1:
                        continue_value = thesplittree.value[1][0, :]
                    else:
                        continue_value = thesplittree.value[1]
                    continue_class_impurity = thesplittree.impurity[1]
                elif splitcondition['right'] == 'continue':
                    continuepath = '>'
                    if thesplittree.n_outputs == 1:
                        continue_value = thesplittree.value[2][0, :]
                    else:
                        continue_value = thesplittree.value[2]
                    continue_class_impurity = thesplittree.impurity[2]
                if continuepath is not None:
                    continue_class_name = ratree.class_names[np.argmax(continue_value)]
                    continue_value = (str(continue_value[0]) + '|' + str(continue_value[1]) +
                                      '|' + str(continue_class_impurity))
                    featuretree.append([splitfeature, continuepath, splitthreshold,
                                        continue_class_name, continue_value])
        # for each in self.featurerulelistdescriptions:
        #     print(each)
        rules_process.save_rules(cfg.get_raw_decision_tree_rules_path(), self.featurerulelistdescriptions)
        # print(cfg.get_raw_decision_tree_rules_path())
        # rules_process.save_rules('decision_tree_rules_raw.txt', self.featurerulelistdescriptions)

    def createthisfeaturetree(self, tree):
        featuretree = []
        for nodeindex in range(1, len(tree)):
            splittree = tree[nodeindex - 1]
            splitfeature = tree[nodeindex].splitmetafeature
            thesplittree = splittree.thetrees[splitfeature].tree_
            splitthreshold = thesplittree.threshold[0]
            splitcondition = splittree.continuesplitconditions[splitfeature]
            splitdecision = None
            continuepath = None
            class_value = None
            continue_value = None
            class_impurity = None
            continue_class_impurity = None
            if splitcondition['left'] == 'D':
                splitdecision = '<='
                if thesplittree.n_outputs == 1:
                    class_value = thesplittree.value[1][0, :]
                else:
                    class_value = thesplittree.value[1]
                class_impurity = thesplittree.impurity[1]
            if splitdecision is not None:
                class_name = ratree.class_names[np.argmax(class_value)]
                class_value = str(class_value[0]) + '|' + str(class_value[1]) + '|' + str(class_impurity)
                featurerulelistdescription = ''
                featurerulelist = list(featuretree)
                featurerulelist.append([splitfeature, splitdecision, splitthreshold, class_name, class_value])
                self.featurerulelists.append(featurerulelist)
                for eachfeaturerule in featurerulelist:
                    if len(featurerulelistdescription) > 0:
                        featurerulelistdescription += ' && '
                    featurerulelistdescription += (str(eachfeaturerule[0]) + eachfeaturerule[1] +
                                                   str(eachfeaturerule[2]) + ':' + str(eachfeaturerule[3]) +
                                                   '|' + str(eachfeaturerule[4]))
                self.featurerulelistdescriptions.append(featurerulelistdescription)

            splitdecision = None
            if splitcondition['right'] == 'D':
                splitdecision = '>'
                if thesplittree.n_outputs == 1:
                    class_value = thesplittree.value[2][0, :]
                else:
                    class_value = thesplittree.value[2]
                class_impurity = thesplittree.impurity[2]
            if splitdecision is not None:
                class_name = ratree.class_names[np.argmax(class_value)]
                class_value = str(class_value[0]) + '|' + str(class_value[1]) + '|' + str(class_impurity)
                featurerulelistdescription = ''
                featurerulelist = list(featuretree)
                featurerulelist.append([splitfeature, splitdecision, splitthreshold, class_name, class_value])
                self.featurerulelists.append(featurerulelist)
                for eachfeaturerule in featurerulelist:
                    if len(featurerulelistdescription) > 0:
                        featurerulelistdescription += ' && '
                    featurerulelistdescription += (str(eachfeaturerule[0]) + eachfeaturerule[1] +
                                                   str(eachfeaturerule[2]) + ':' + str(eachfeaturerule[3]) +
                                                   '|' + str(eachfeaturerule[4]))
                self.featurerulelistdescriptions.append(featurerulelistdescription)

            if splitcondition['left'] == 'continue':
                continuepath = '<='
                if thesplittree.n_outputs == 1:
                    continue_value = thesplittree.value[1][0, :]
                else:
                    continue_value = thesplittree.value[1]
                continue_class_impurity = thesplittree.impurity[1]
            elif splitcondition['right'] == 'continue':
                continuepath = '>'
                if thesplittree.n_outputs == 1:
                    continue_value = thesplittree.value[2][0, :]
                else:
                    continue_value = thesplittree.value[2]
                continue_class_impurity = thesplittree.impurity[2]
            if continuepath is not None:
                continue_class_name = ratree.class_names[np.argmax(continue_value)]
                continue_value = (str(continue_value[0]) + '|' + str(continue_value[1]) +
                                  '|' + str(continue_class_impurity))
                featuretree.append([splitfeature, continuepath, splitthreshold,
                                    continue_class_name, continue_value])

    def __init__(self, train_data):
        self.trees = []
        self.pairtable = train_data
        pairtablevalues = self.pairtable.values
        self.newX = np.array(pairtablevalues[0:, 2:]).astype(np.float32)
        self.newY = pairtablevalues[0:, 1].reshape(-1, 1).astype(np.int32)
        self.featurenames = self.pairtable.columns.tolist()[2:]
        self.currentdepth = 1
        self.featurenameindex = {}
        for index in range(0, len(self.featurenames)):
            self.featurenameindex[self.featurenames[index]] = index
        self.continuesplitconditions = {}
        self.thetrees = {}
        self.recursivepath = [self]
        self.featuretrees = []
        self.featurerulelists = []
        self.featurerulelistdescriptions = []

        total_features_size = len(self.featurenames)
        count = 1
        for eachmetafeature in self.featurenames:
            start_time = time.time()
            print('- running feature ({}/{}): {}'.format(count, total_features_size, eachmetafeature))
            nextsplitmetafeatureindex = self.featurenames.index(eachmetafeature)
            newX = self.newX[:, nextsplitmetafeatureindex].reshape(-1, 1)

            # 1st classweight
            clf = tree.DecisionTreeClassifier(max_depth=1,
                                              class_weight={0: set_class_weight, 1: 1})
            clf = clf.fit(newX, self.newY)
            self.thetrees[eachmetafeature] = clf
            if len(clf.tree_.impurity) > 1:
                leftimpurity = clf.tree_.impurity[1]
                rightimpurity = clf.tree_.impurity[2]
                left_class_value = None
                right_class_value = None
                left_class_name = None
                right_class_name = None
                left_treestopnormalizedthreshold = None
                right_treestopnormalizedthreshold = None
                if clf.tree_.n_outputs == 1:
                    left_class_value = clf.tree_.value[1][0, :]
                    right_class_value = clf.tree_.value[2][0, :]
                else:
                    left_class_value = clf.tree_.value[1]
                    right_class_value = clf.tree_.value[2]

                #####wang write
                left_class_value[0] = left_class_value[0] / set_class_weight
                right_class_value[0] = right_class_value[0] / set_class_weight
                leftimpurity = 1 - pow(left_class_value[0] / (left_class_value[0] + left_class_value[1]), 2) - pow(
                    left_class_value[1] / (left_class_value[0] + left_class_value[1]), 2)
                rightimpurity = 1 - pow(right_class_value[0] / (right_class_value[0] + right_class_value[1]), 2) - pow(
                    right_class_value[1] / (right_class_value[0] + right_class_value[1]), 2)
                #####
                clf.tree_.impurity[1] = leftimpurity
                clf.tree_.impurity[2] = rightimpurity
                left_class_name = ratree.class_names[np.argmax(left_class_value)]
                right_class_name = ratree.class_names[np.argmax(right_class_value)]
                if left_class_name == 'U':
                    left_treestopnormalizedthreshold = unmatch_tree_split_threshold
                else:
                    left_treestopnormalizedthreshold = match_tree_split_threshold
                if right_class_name == 'U':
                    right_treestopnormalizedthreshold = unmatch_tree_split_threshold
                else:
                    right_treestopnormalizedthreshold = match_tree_split_threshold
                if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                    self.continuesplitconditions[eachmetafeature] = {'left': 'finalize', 'right': 'finalize'}
                else:
                    if leftimpurity < left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                        self.continuesplitconditions[eachmetafeature] = {'left': 'D', 'right': 'continue'}
                    else:
                        if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                            self.continuesplitconditions[eachmetafeature] = {'left': 'continue', 'right': 'D'}
                        else:
                            if leftimpurity < left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                                self.continuesplitconditions[eachmetafeature] = {'left': 'D', 'right': 'D'}
                ratree.tree(self, self, eachmetafeature, list(self.recursivepath))

            # 2nd classweight
            clf = tree.DecisionTreeClassifier(max_depth=1,
                                              class_weight={0: 1, 1: set_class_weight})
            clf = clf.fit(newX, self.newY)
            self.thetrees[eachmetafeature] = clf
            if len(clf.tree_.impurity) > 1:
                leftimpurity = clf.tree_.impurity[1]
                rightimpurity = clf.tree_.impurity[2]
                left_class_value = None
                right_class_value = None
                left_class_name = None
                right_class_name = None
                left_treestopnormalizedthreshold = None
                right_treestopnormalizedthreshold = None
                if clf.tree_.n_outputs == 1:
                    left_class_value = clf.tree_.value[1][0, :]
                    right_class_value = clf.tree_.value[2][0, :]
                else:
                    left_class_value = clf.tree_.value[1]
                    right_class_value = clf.tree_.value[2]

                #####wang write
                left_class_value[1] = left_class_value[1] / set_class_weight
                right_class_value[1] = right_class_value[1] / set_class_weight
                leftimpurity = 1 - pow(left_class_value[0] / (left_class_value[0] + left_class_value[1]), 2) - pow(
                    left_class_value[1] / (left_class_value[0] + left_class_value[1]), 2)
                rightimpurity = 1 - pow(right_class_value[0] / (right_class_value[0] + right_class_value[1]), 2) - pow(
                    right_class_value[1] / (right_class_value[0] + right_class_value[1]), 2)

                #####
                clf.tree_.impurity[1] = leftimpurity
                clf.tree_.impurity[2] = rightimpurity
                left_class_name = ratree.class_names[np.argmax(left_class_value)]
                right_class_name = ratree.class_names[np.argmax(right_class_value)]
                if left_class_name == 'U':
                    left_treestopnormalizedthreshold = unmatch_tree_split_threshold
                else:
                    left_treestopnormalizedthreshold = match_tree_split_threshold
                if right_class_name == 'U':
                    right_treestopnormalizedthreshold = unmatch_tree_split_threshold
                else:
                    right_treestopnormalizedthreshold = match_tree_split_threshold
                if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                    self.continuesplitconditions[eachmetafeature] = {'left': 'finalize', 'right': 'finalize'}
                else:
                    if leftimpurity < left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                        self.continuesplitconditions[eachmetafeature] = {'left': 'D', 'right': 'continue'}
                    else:
                        if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                            self.continuesplitconditions[eachmetafeature] = {'left': 'continue', 'right': 'D'}
                        else:
                            if leftimpurity < left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                                self.continuesplitconditions[eachmetafeature] = {'left': 'D', 'right': 'D'}
                ratree.tree(self, self, eachmetafeature, list(self.recursivepath))
            count += 1
            print("- running time: {}s.".format(time.time() - start_time))

    class tree:
        featurenames = None
        currentdepth = None
        X = None
        Y = None
        newX = None
        newY = None
        roottree = None
        uptree = None
        splitmetafeature = None
        splitmetafeatureindex = None
        splitpairindexes = None
        splitpairvalues = None
        continuesplitconditions = None
        thetrees = None
        recursivepath = None

        def __init__(self, roottree, uptree, splitmetafeature, recursivepath):
            self.roottree = roottree
            self.uptree = uptree
            self.splitmetafeature = splitmetafeature
            self.currentdepth = uptree.currentdepth + 1
            self.featurenames = list(uptree.featurenames)
            self.splitmetafeatureindex = self.featurenames.index(self.splitmetafeature)
            if 'diff' in self.splitmetafeature:
                self.featurenames.remove(self.splitmetafeature)
            self.continuesplitconditions = {}
            self.thetrees = {}
            self.recursivepath = recursivepath
            self.recursivepath.append(self)
            finalize = True
            if self.currentdepth <= max_tree_depth:
                self.X = np.array(uptree.newX)
                self.Y = np.array(uptree.newY)
                self.splitpairvalues = self.X[:, self.splitmetafeatureindex]
                if 'diff' in self.splitmetafeature:
                    self.X = np.delete(self.X, self.splitmetafeatureindex, 1)
                self.splitpairindexes = []
                if uptree.continuesplitconditions[splitmetafeature]['left'] == 'continue':
                    finalize = False
                    for index in range(0, self.Y.size):
                        if self.splitpairvalues[index] <= uptree.thetrees[splitmetafeature].tree_.threshold[0]:
                            self.splitpairindexes.append(index)
                else:
                    if uptree.continuesplitconditions[splitmetafeature]['right'] == 'continue':
                        finalize = False
                        for index in range(0, self.Y.size):
                            if self.splitpairvalues[index] > uptree.thetrees[splitmetafeature].tree_.threshold[0]:
                                self.splitpairindexes.append(index)
            if finalize == True:
                # if uptree.continuesplitconditions[splitmetafeature]['right'] == 'D' or uptree.continuesplitconditions[splitmetafeature]['left'] == 'D':
                # self.roottree.trees.append(self.recursivepath)
                self.roottree.createthisfeaturetree(self.recursivepath)
            else:
                self.newX = self.X[self.splitpairindexes]
                self.newY = self.Y[self.splitpairindexes]
                for nextsplitfeature in self.featurenames:
                    # print('#################tree ##self.featurenames:',nextsplitfeature)
                    nextsplitmetafeatureindex = self.featurenames.index(nextsplitfeature)
                    newX = self.newX[:, nextsplitmetafeatureindex].reshape(-1, 1)

                    # 1st classweight
                    clf = tree.DecisionTreeClassifier(max_depth=1,
                                                      class_weight={0: set_class_weight, 1: 1})
                    clf = clf.fit(newX, self.newY)
                    self.thetrees[nextsplitfeature] = clf
                    if len(clf.tree_.impurity) == 1:
                        # self.roottree.trees.append(self.recursivepath)
                        self.roottree.createthisfeaturetree(self.recursivepath)
                    else:
                        leftimpurity = clf.tree_.impurity[1]
                        rightimpurity = clf.tree_.impurity[2]
                        left_class_value = None
                        right_class_value = None
                        left_class_name = None
                        right_class_name = None
                        left_treestopnormalizedthreshold = None
                        right_treestopnormalizedthreshold = None
                        if clf.tree_.n_outputs == 1:
                            left_class_value = clf.tree_.value[1][0, :]
                            right_class_value = clf.tree_.value[2][0, :]
                        else:
                            left_class_value = clf.tree_.value[1]
                            right_class_value = clf.tree_.value[2]

                        #####wang write
                        left_class_value[0] = left_class_value[0] / set_class_weight
                        right_class_value[0] = right_class_value[0] / set_class_weight
                        leftimpurity = 1 - pow(left_class_value[0] / (left_class_value[0] + left_class_value[1]),
                                               2) - pow(
                            left_class_value[1] / (left_class_value[0] + left_class_value[1]), 2)
                        rightimpurity = 1 - pow(right_class_value[0] / (right_class_value[0] + right_class_value[1]),
                                                2) - pow(
                            right_class_value[1] / (right_class_value[0] + right_class_value[1]), 2)
                        #####
                        clf.tree_.impurity[1] = leftimpurity
                        clf.tree_.impurity[2] = rightimpurity

                        left_class_name = ratree.class_names[np.argmax(left_class_value)]
                        right_class_name = ratree.class_names[np.argmax(right_class_value)]
                        if left_class_name == 'U':
                            left_treestopnormalizedthreshold = unmatch_tree_split_threshold
                        else:
                            left_treestopnormalizedthreshold = match_tree_split_threshold
                        if right_class_name == 'U':
                            right_treestopnormalizedthreshold = unmatch_tree_split_threshold
                        else:
                            right_treestopnormalizedthreshold = match_tree_split_threshold
                        if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                            self.continuesplitconditions[nextsplitfeature] = {'left': 'finalize', 'right': 'finalize'}
                        else:
                            if leftimpurity < left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                                self.continuesplitconditions[nextsplitfeature] = {'left': 'D', 'right': 'continue'}
                            else:
                                if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                                    self.continuesplitconditions[nextsplitfeature] = {'left': 'continue', 'right': 'D'}
                                else:
                                    if leftimpurity < left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                                        self.continuesplitconditions[nextsplitfeature] = {'left': 'D', 'right': 'D'}
                        ratree.tree(self.roottree, self, nextsplitfeature, list(self.recursivepath))

                    # 2nd classweight
                    clf = tree.DecisionTreeClassifier(max_depth=1,
                                                      class_weight={0: 1, 1: set_class_weight})
                    clf = clf.fit(newX, self.newY)
                    self.thetrees[nextsplitfeature] = clf
                    if len(clf.tree_.impurity) == 1:
                        # self.roottree.trees.append(self.recursivepath)
                        self.roottree.createthisfeaturetree(self.recursivepath)
                    else:
                        leftimpurity = clf.tree_.impurity[1]
                        rightimpurity = clf.tree_.impurity[2]
                        left_class_value = None
                        right_class_value = None
                        left_class_name = None
                        right_class_name = None
                        left_treestopnormalizedthreshold = None
                        right_treestopnormalizedthreshold = None
                        if clf.tree_.n_outputs == 1:
                            left_class_value = clf.tree_.value[1][0, :]
                            right_class_value = clf.tree_.value[2][0, :]
                        else:
                            left_class_value = clf.tree_.value[1]
                            right_class_value = clf.tree_.value[2]

                        #####wang write
                        left_class_value[1] = left_class_value[1] / set_class_weight
                        right_class_value[1] = right_class_value[1] / set_class_weight
                        leftimpurity = 1 - pow(left_class_value[0] / (left_class_value[0] + left_class_value[1]),
                                               2) - pow(
                            left_class_value[1] / (left_class_value[0] + left_class_value[1]), 2)
                        rightimpurity = 1 - pow(right_class_value[0] / (right_class_value[0] + right_class_value[1]),
                                                2) - pow(
                            right_class_value[1] / (right_class_value[0] + right_class_value[1]), 2)
                        #####
                        clf.tree_.impurity[1] = leftimpurity
                        clf.tree_.impurity[2] = rightimpurity

                        left_class_name = ratree.class_names[np.argmax(left_class_value)]
                        right_class_name = ratree.class_names[np.argmax(right_class_value)]
                        if left_class_name == 'U':
                            left_treestopnormalizedthreshold = unmatch_tree_split_threshold
                        else:
                            left_treestopnormalizedthreshold = match_tree_split_threshold
                        if right_class_name == 'U':
                            right_treestopnormalizedthreshold = unmatch_tree_split_threshold
                        else:
                            right_treestopnormalizedthreshold = match_tree_split_threshold
                        if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                            self.continuesplitconditions[nextsplitfeature] = {'left': 'finalize', 'right': 'finalize'}
                        else:
                            if leftimpurity < left_treestopnormalizedthreshold and rightimpurity >= right_treestopnormalizedthreshold:
                                self.continuesplitconditions[nextsplitfeature] = {'left': 'D', 'right': 'continue'}
                            else:
                                if leftimpurity >= left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                                    self.continuesplitconditions[nextsplitfeature] = {'left': 'continue', 'right': 'D'}
                                else:
                                    if leftimpurity < left_treestopnormalizedthreshold and rightimpurity < right_treestopnormalizedthreshold:
                                        self.continuesplitconditions[nextsplitfeature] = {'left': 'D', 'right': 'D'}
                        ratree.tree(self.roottree, self, nextsplitfeature, list(self.recursivepath))

def generate_rules():
    # global match_tree_split_threshold, unmatch_tree_split_threshold, match_rule_threshold, unmatch_rule_threshold
    # All data information
    rule_out_info = open(cfg.get_info_decision_tree_rules_path(), 'w')
    rule_out_info.write("### {} ###".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    df = pd.read_csv(cfg.get_raw_data_path())
    pairs = df.values
    # print("- # of data: {}.".format(len(pairs)))
    id_2_pair_info = dict()
    for elem in pairs:
        id_2_pair_info[elem[0]] = elem  # <id, info>

    # Train data information
    train_data_info = []
    train_info = pd.read_csv(cfg.train).values
    train_ids = train_info[:, 0].astype(str)  ######wang delete
    # train_ids = pairs[:, 0].astype(str)   ###wang add
    msg = "- # of train data: {}".format(len(train_ids))
    # print(msg)
    rule_out_info.write(msg + '\n')
    for train_id in train_ids:
        train_data_info.append(id_2_pair_info.get(train_id))
    train_data_info_df = pd.DataFrame(train_data_info, columns=df.columns)
    # train_data_info_df.to_csv('hello.csv', index=False)
    train_labels = np.array(train_data_info)[:, 1]
    label_counts = Counter(train_labels)
    train_impurity = 1.0
    for k, v in label_counts.items():
        train_impurity -= (1.0 * v / len(train_labels)) ** 2
    msg = "- class size of train data: {}, impurity of train data: {}".format(label_counts, train_impurity)
    # print(msg)
    rule_out_info.write(msg + '\n')

    # Set threshold according to the impurity of training data.
    # match_tree_split_threshold = 0.1
    # unmatch_tree_split_threshold = 0.01
    # match_tree_split_threshold = 0.3
    # unmatch_tree_split_threshold = 0.015
    # match_rule_threshold = match_tree_split_threshold
    # unmatch_rule_threshold = unmatch_tree_split_threshold

    # Print configuration
    msg = ''
    msg += '- max_tree_depth: {}\n'.format(max_tree_depth)
    # msg += '- min_samples_leaf: {}\n'.format(min_samples_leaf)
    msg += '- match_tree_split_threshold: {}\n'.format(match_tree_split_threshold)
    msg += '- unmatch_tree_split_threshold: {}\n'.format(unmatch_tree_split_threshold)
    msg += '- match_rule_threshold: {}\n'.format(match_rule_threshold)
    msg += '- unmatch_rule_threshold: {}\n'.format(unmatch_rule_threshold)
    # print(msg)
    rule_out_info.write(msg)

    # Generate rules using decision tree.
    ratree_ = ratree(train_data_info_df)
    ratree_.createfeaturetrees()

    # Read rules and clean them.
    read_rules = rules_process.read_rules(cfg.get_raw_decision_tree_rules_path())
    # -- select low impurity rules;
    cleaned_rules = rules_process.select_rules_based_on_threshold(read_rules,
                                                                  match_rule_threshold,
                                                                  unmatch_rule_threshold)
    # -- deduplicate rules;
    # cleaned_rules = rules_process.clean_rules(cleaned_rules, True)
    # multiple processes version.
    cleaned_rules = rules_process.clean_rules_mt(cleaned_rules, 5)
    msg = "- (Before cleaning) # of rules: {}, Classes: {}".format(
        len(read_rules), Counter([elem.infer_class for elem in read_rules]))
    # print(msg)
    rule_out_info.write(msg + '\n')
    msg = "- (After cleaning) # of rules: {}, Classes: {}".format(
        len(cleaned_rules), Counter([elem.infer_class for elem in cleaned_rules]))
    # print(msg)
    rule_out_info.write(msg + '\n')

    # Store cleaned rules.
    clean_rule_text = [elem.original_description for elem in cleaned_rules]
    rules_process.save_rules(cfg.get_decision_tree_rules_path(), clean_rule_text)

    # Output readable cleaned rules according to their classes.
    match_rules = []
    unmatch_rules = []
    for elem in cleaned_rules:
        if elem.infer_class == 'M':
            match_rules.append(elem.readable_description)
        else:
            unmatch_rules.append(elem.readable_description)
    rule_out_info.write('\n--- Match rules ---\n')
    for elem in match_rules:
        rule_out_info.write(elem + '\n')
    rule_out_info.write('\n--- Unmatch rules ---\n')
    for elem in unmatch_rules:
        rule_out_info.write(elem + '\n')
    rule_out_info.write("### The End (%s). ###" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    rule_out_info.flush()
    rule_out_info.close()


def evaluate_rule_generation_scalability():
    import datetime
    # All data information
    path_suffix = cfg.get_risk_features_path().split('/')[-3] + '-rule-scalability.txt'
    runtime_info = open(cfg.get_risk_features_path() + path_suffix, 'w')
    df = pd.read_csv(cfg.get_raw_data_path())
    pairs = df.values
    print("- # of data: {}.".format(len(pairs)))
    id_2_pair_info = dict()
    for elem in pairs:
        id_2_pair_info[elem[0]] = elem  # <id, info>

    # Train data information
    train_data_info = []
    train_info = pd.read_csv(cfg.get_risk_features_path() + 'train.csv').values
    train_ids = train_info[:, 0].astype(str)
    total_training_size = len(train_ids)
    msg = "- # of total training data: {}".format(total_training_size)
    print(msg)
    for train_id in train_ids:
        train_data_info.append(id_2_pair_info.get(train_id))
    train_data_info_df = pd.DataFrame(train_data_info, columns=df.columns)

    set_training_size = 0
    delta_size = 1000
    while set_training_size < total_training_size:
        set_training_size += delta_size
        set_training_size = np.minimum(set_training_size, total_training_size)
        current_train_data_info_df = train_data_info_df.iloc[:set_training_size, :]
        print("- Current # of training data: {}".format(set_training_size))
        msg = '{}, '.format(set_training_size)
        msg += '{}, '.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # Generate rules using decision tree.
        ratree_ = ratree(current_train_data_info_df)
        ratree_.createfeaturetrees()
        msg += '{}(DTree Done.), '.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        # Read rules and clean them.
        read_rules = rules_process.read_rules(cfg.get_raw_decision_tree_rules_path())
        # -- select low impurity rules;
        cleaned_rules = rules_process.select_rules_based_on_threshold(read_rules,
                                                                      match_rule_threshold,
                                                                      unmatch_rule_threshold)
        # -- deduplicate rules;
        cleaned_rules = rules_process.clean_rules_mt(cleaned_rules, 5)
        msg += '{}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        runtime_info.write(msg)

def generate_all_rules(input_path, output_path, class_num, macth_gini, unmatch_gini, max_depth):
    global match_tree_split_threshold
    global match_rule_threshold
    global unmatch_tree_split_threshold
    global unmatch_rule_threshold
    global max_tree_depth

    ############################ ?
    match_tree_split_threshold = macth_gini
    match_rule_threshold = macth_gini
    unmatch_tree_split_threshold = unmatch_gini
    unmatch_rule_threshold = unmatch_gini
    max_tree_depth = max_depth

    for i in tqdm(range(class_num), desc="decision tree to rules"):
        cfg.raw_data_path = os.path.join(input_path, str(i) + '_all_data_info.csv')

        cfg.raw_decision_tree_rules_path = os.path.join(output_path, str(i) + '_decision_tree_rules_raw.txt')
        cfg.decision_tree_rules_path = os.path.join(output_path, str(i) + '_decision_tree_rules_clean.txt')
        cfg.info_decision_tree_rules_path = os.path.join(output_path, str(i) + '_decision_tree_rules_info.txt')
        cfg.train = os.path.join(input_path, str(i) + '_train.csv')
        generate_rules()


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

def prepare_pair_info_more(data_name, class_num):
    from copy import copy
    from numpy import transpose
    path = '{}/{}/all_data_info.csv'.format(data_name, rate)

    df_pair_info_more = pd.read_csv(path, header=None)
    pair_info_more = df_pair_info_more.values
    pair_info_more = transpose(pair_info_more)

    path = '{}/{}/train.csv'.format(data_name, rate)
    df_train = pd.read_csv(path, header=None)
    train = df_train.values
    train = transpose(train)

    for index in range(class_num):
        c = int(index)

        cur_pair = []
        for line in pair_info_more:
            if line[0] == 'data' or line[0] == 'label':
                cur_pair.append(copy(line))
            else:
                temp = line[0].split('_')
                if int(temp[3]) == c:
                    cur_pair.append(copy(line))
        cur_pair = transpose(cur_pair)
        for i in range(1, len(cur_pair)):
            if int(cur_pair[i][1]) == c:
                cur_pair[i][1] = 1
            else:
                cur_pair[i][1] = 0


        cur_train = []
        for line in train:
            if line[0] == 'data' or line[0] == 'label':
                cur_train.append(copy(line))
            else:
                temp = line[0].split('_')
                if int(temp[3]) == c:
                    cur_train.append(copy(line))
        cur_train = transpose(cur_train)
        for i in range(1, len(cur_train)):
            if int(cur_train[i][1]) == c:
                cur_train[i][1] = 1
            else:
                cur_train[i][1] = 0

        path = "./{}/{}".format(data_name, rate)
        if not os.path.exists(path):
            os.makedirs(path)

        pd.DataFrame(cur_pair).to_csv("./{}/{}/{}_all_data_info.csv".format(data_name, rate, c), header=None, index=None)
        pd.DataFrame(cur_train).to_csv("./{}/{}/{}_train.csv".format(data_name, rate, c), header=None, index=None)




        # # 保留所有属性
        # c = int(index)
        # cur_pair = copy(pair_info_more)
        # cur_train = copy(train)
        # for i in range(len(cur_pair)):
        #     if int(cur_pair[i][1]) == c:
        #         cur_pair[i][1] = 1
        #     else:
        #         cur_pair[i][1] = 0
        #
        # for i in range(len(cur_train)):
        #     if int(cur_train[i][1]) == c:
        #         cur_train[i][1] = 1
        #     else:
        #         cur_train[i][1] = 0
        # cur_pair = np.vstack((head1, cur_pair))
        # cur_train = np.vstack((head2, cur_train))
        # pd.DataFrame(cur_pair).to_csv("./{}/{}_all_data_info.csv".format(data_name, c), header=None, index=None)
        # pd.DataFrame(cur_train).to_csv("./{}/{}_train.csv".format(data_name, c), header=None, index=None)

    print()

import time

if __name__ == '__main__':

    start = time.time()

    prepare_pair_info_more(data_path, class_num)

    input_path = "./{}/{}".format(data_path, rate)
    output_path = "./{}_output/{}".format(data_name, rate)

    # mkdir
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    generate_all_rules(input_path, output_path, class_num, macth_gini, unmatch_gini, max_depth)

    rules_merge(output_path, output_path, class_num)

    end = time.time()

    # running_time = end - start
    # print("time : ",end - start)

    print('time cost : %.2f sec' % (end - start))
