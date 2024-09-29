import torch
from PrepareRiskDataset.config.overall_config import *

# 基本信息
dataset = source_dataset  # train model path
data_name = f'{dataset}_bert' # model path
fine_tune_dataset = target_dataset   # fine tune dataset
data_path = f"{source_dataset}_bert/{rate}/"

cpu = torch.device('cpu')
gpu = torch.device('cuda:0')
maxlen = 150
batch_size = 8

best_model = 'best_model.weights'  # 'best_model.h5'(bert4keras), 'best_model.weights'（bert）, 'bestmodel.pt'(textcnn), transformers不用加载
code_frame = "bert4keras"  # pytorch(textcnn), bert4keras(bert), keras_bert
model_name = 'bert'  # 'BertGCN',bert, if textcnn, need reviews args in get_CNN_feature

if model_name=="bert":
    get_layer = 'Pooler-Dense'
    get_layer1 = 'classify_output'