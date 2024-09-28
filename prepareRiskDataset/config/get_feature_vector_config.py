import torch

# 基本信息
cpu = torch.device('cpu')
gpu = torch.device('cuda:1')
maxlen = 150
# epochs = 10
batch_size = 8
# learning_rate = 4e-5
# crf_lr_multiplier = 100  # 必要时扩大CRF层的学习率

################################# dataset information
# classify_num_labels = 20
# data_name = 'fudan_bert'
# rate = 1.0

# classify_num_labels = 14
# data_name = 'qinghua_bert'
# rate = 1.0

# classify_num_labels = 26
# data_name = 'NYT_bert'
# rate = 0.4



# classify_num_labels = 3
# data_name = 'jiaocha_bert'
# cur_path = './' + data_name + '/'

###################3 model layers
dataset = "AgNews"  # train model path
data_name = f'{dataset}_bert' # model path
fine_tune_dataset = "BBC"   # fine tune dataset
classify_num_labels = 4
rate = 1.0
data_path = f"{fine_tune_dataset}_bert/{rate}/"

best_model = 'bestmodel.pt'  # 'best_model.h5'(bert4keras), 'best_model.weights'（bert）, 'bestmodel.pt'(textcnn), transformers不用加载
code_frame = "pytorch"  # pytorch(textcnn), bert4keras(bert), keras_bert
model_name = 'textcnn'  # 'BertGCN',bert, if textcnn, need reviews args in get_CNN_feature

if model_name=="bert":
    get_layer = 'Pooler-Dense'
    get_layer1 = 'classify_output'
elif model_name=="bertgcn":
    get_layer = 'bert_model'
    get_layer1 = 'classifier'

# cur_path = './' + data_name + '/'
# cur_path = 'E:\\study\\研究生毕设\\code\\risk\\datasets\\20News\\risk_dataset-0.4\\'

###### pytorch
# if model_name=="bert":
#     # config_path = 'bert-base/bert_config.json'
#     # checkpoint_path = 'bert-base/bert_model.ckpt'
#     dict_path = 'bert-base'

# #   bert4keras config
if data_name == 'fudan_bert' or data_name == 'qinghua_bert' or data_name == 'jiaocha_bert':
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'
else:
    config_path = 'uncased_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'