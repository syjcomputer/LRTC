import torch

# overall configure
classify_num_labels = 4
source_dataset = 'AgNews'
target_dataset = 'BBC'
rate = 1.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vector_cache = "pretrained_vector"
if source_dataset == 'fudan_bert' or source_dataset == 'qinghua_bert' or source_dataset == 'jiaocha_bert':
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
else:
    config_path = 'uncased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

# chi_new.py config
p = 0.1
feature_per_class = 100
total_feature =  classify_num_labels * feature_per_class

# get_risk_dataset.py
cnns = ['bert',"cnn"] # ['cnn', 'att']
layers = ["net", "dense"] #
elems = ["distance", 'knn5', 'knn8'] # ["distance", 'knn1', 'knn8']
token_name = ['chi'] # 'tfidf', "chi"
k_list = [5, 8] # [1, 8]