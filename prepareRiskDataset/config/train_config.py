import torch

# 基本信息
maxlen = 150
epochs = 30
batch_size = 64
learning_rate = 4e-5  # 4e-5
crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率
dataset = "try_data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# classify_num_labels = 20
# data_name = 'fudan_bert'
# rate = 1.0

# classify_num_labels = 26
# data_name = 'NYT_bert'

classify_num_labels = 4
data_name = f'{dataset}_bert'
rate = 1.0

# classify_num_labels = 14
# data_name = 'qinghua_bert'
# rate = 1.0


cur_path = './{}/{}/'.format(data_name, rate)
model_name = "bert"

num_cores = 10
# classify_num_labels = 3
# data_name = 'jiaocha_bert'
# cur_path = './' + data_name + '/'

# bertgcn
'''bertgcn_min_count=5
bertgcn_window_size=20
bertgcn_word_embeddings_dim=300
gcn_model = 'gcn'
bert_init = 'roberta-base' # 'roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'
m = 0.7 # the factor balancing BERT and GCN prediction
gcn_layers = 2
gcn_n_hidden = 200
gcn_dropout = 0.5
gat_heads = 8'''


if model_name=="bert":
    if data_name == 'fudan_bert' or data_name == 'qinghua_bert' or data_name == 'jiaocha_bert':
        config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
    else:
        config_path = 'uncased_L-12_H-768_A-12/bert_config.json'
        checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
        dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'
# if model_name=="bert":
#     # config_path = 'bert-base/bert_config.json'
#     # checkpoint_path = 'bert-base/bert_model.ckpt'
#     dict_path = 'bert-base'

elif model_name=="roberta":
    dict_path = "/home/ssd1/syj/risk_code/zh_get_risk_dataset/xlnet"
    # if data_name == 'fudan_bert' or data_name == 'qinghua_bert' or data_name == 'jiaocha_bert':
    #     pass
    # else:
    #     config_path = 'tf_roberta_base/bert_config.json'
    #     checkpoint_path = 'tf_roberta_base/tf_roberta_base.ckpt'
    #     dict_path = 'tf_roberta_base/dict.txt'