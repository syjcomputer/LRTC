# **[TransferLearningViaModelRisk](https://github.com/syjcomputer/TransferLearningViaModelRisk)**

本项目为在少量样本情况下借用风险分析模型微调预训练模型，主要针对新闻分类任务的迁移学习任务。

风险分析框架如下图：

![risklearn](D:\risk_for_text_cls_for_git\risklearn.png)

迁移学习对比结果如下：

|           | 20News-AgNews | AgNews-20News | AgNews-BBC | BBC-AgNews | BBC-20News | 20News-BBC |
| --------- | ------------- | ------------- | ---------- | ---------- | ---------- | ---------- |
| Bert      | 91.2          | 83.1          | 89.48      | 78.92      | 62.68      | 88.75      |
| Roberta   | 77.38         | 79.47         | 65.87      | 75.06      | 54.20      | 61.8       |
| XLnet     | 82.87         | 86.23         | 82.77      | 63.05      | 58.87      | 81.06      |
| TextCNN   | 43.78         | 61.05         | 64.39      | 41.23      | 54.64      | 38.79      |
| BertGCN   | 36.06         | 54.97         | 35.36      | 36.38      | 52.4       | 40.04      |
| BERT+CAT  | 31.2          | 50.39         | 36.42      | 35.61      | 30.7       | 44.5       |
| npc-gzip  | 43.51         | 79.05         | 59.22      | 32.24      | 54.76      | 46.45      |
| BERT+UST  | 90.96         | 83.4          | 92.12      | 73.13      | 69.26      | 70.93      |
| NSP       | 81.94         | 85.87         | 77.64      | 75.79      | **70.13**  | 91.69      |
| LearnRisk | **92.11**     | **86.32**     | **93.92**  | **78.94**  | 68.19      | **93.22**  |


## Installation

```
python==3.7
torch==1.8.0+cu11
tensorflow-gpu==2.2.2
keras==2.3.1
bert4keras==0.11.4
numpy==1.18.5
scikit-learn==1.0.2
```

## Quick Start

### 文档结构



### 数据准备



### 模型准备

### 运行顺序

```
zh_get_risk_dataset.get_risk_dataset.py -> 

OneSideDecisionTree.zh_decision_tree_2rules.py -> 

risk.zh_train-diedai.py
```

## Citations

```bibtex

```

