# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from PrepareRiskDataset.train_util import *
from bert4keras.snippets import sequence_padding
import jieba
import torchtext
from torchtext.data import Field,TabularDataset,Iterator,BucketIterator,LabelField
from PrepareRiskDataset.config.train_config import *


def stopwordslist(path):
    stopwords = [line.strip() for line in open(path, encoding='UTF-8').readlines()]
    return stopwords

class TextCNN2(nn.Module):
    def __init__(self, args):
        super(TextCNN2, self).__init__()
        self.args = args

        label_num = args.label_num # 标签的个数
        filter_num = args.filter_num # 卷积核的个数
        filter_sizes = [int(fsz) for fsz in args.filter_sizes]

        vocab_size = args.vocab_size
        embedding_dim = args.embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if args.static: # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(args.vectors, freeze=not args.fine_tune)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (fsz, embedding_dim)) for fsz in filter_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(len(filter_sizes)*filter_num, label_num)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大=长度
        x = self.embedding(x) # 经过embedding,x的维度为(batch_size, max_len, embedding_dim)

        # 经过view函数x的维度变为(batch_size, input_chanel=1, w=max_len, h=embedding_dim)
        x = x.view(x.size(0), 1, x.size(1), self.args.embedding_dim)

        # 经过卷积运算,x中每个运算结果维度为(batch_size, out_chanel, w, h=1)
        x = [F.relu(conv(x)) for conv in self.convs]

        # 经过最大池化层,维度变为(batch_size, out_chanel, w=1, h=1)
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]

        # 将不同卷积核运算结果维度（batch，out_chanel,w,h=1）展平为（batch, outchanel*w*h）
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        # 将不同卷积核提取的特征组合起来,维度变为(batch, sum:outchanel*w*h)
        x = torch.cat(x, 1)
        # dropout层
        x = self.dropout(x)
        # 全连接层
        logits = self.linear(x)
        return logits

class TextCNN(nn.Module):
    def __init__(self, args, vectors,maxlen=150): # dropout率
        super(TextCNN, self).__init__() # 继承nn.Module

        chanel_num = 1  # 通道数，也就是一篇文章一个样本只相当于一个feature map

        self.maxlen = maxlen

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim) # 嵌入层

        if args.static: # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(vectors) #嵌入层加载预训练词向量

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, args.filter_num, (fsz, args.embedding_dim)) for fsz in args.filter_sizes])  # 卷积层
        self.dropout = nn.Dropout(args.dropout) # dropout
        self.fc = nn.Linear(len(args.filter_sizes) * args.filter_num, args.label_num) #全连接层

    def forward(self, x):
        x = x.permute(1,0)
        x = torch.tensor(sequence_padding(x.cpu().numpy(), self.maxlen)).to(device)
        # x维度[句子长度,一个batch中所包含的样本数] 例:[3451,128]
        x = self.embedding(x) # #经过嵌入层之后x的维度，[句子长度,一个batch中所包含的样本数,词向量维度] 例：[3451,128,300]
        #x = x.permute(1,0,2) # permute函数将样本数和句子长度换一下位置，[一个batch中所包含的样本数,句子长度,词向量维度] 例：[128,3451,300]
        x = x.unsqueeze(1) # # conv2d需要输入的是一个四维数据，所以新增一维feature map数 unsqueeze(1)表示在第一维处新增一维，[一个batch中所包含的样本数,一个样本中的feature map数，句子长度,词向量维度] 例：[128,1,3451,300]
        x = [conv(x) for conv in self.convs] # 与卷积核进行卷积，输出是[一个batch中所包含的样本数,卷积核数，句子长度-卷积核size+1,1]维数据,因为有[3,4,5]三张size类型的卷积核所以用列表表达式 例：[[128,16,3459,1],[128,16,3458,1],[128,16,3457,1]]
        x = [sub_x.squeeze(3) for sub_x in x]#squeeze(3)判断第三维是否是1，如果是则压缩，如不是则保持原样 例：[[128,16,3459],[128,16,3458],[128,16,3457]]
        x = [F.relu(sub_x) for sub_x in x] # ReLU激活函数激活，不改变x维度

        x = [F.max_pool1d(sub_x,sub_x.size(2)) for sub_x in x] # 池化层，根据之前说的原理，max_pool1d要取出每一个滑动窗口生成的矩阵的最大值，因此在第二维上取最大值 例：[[128,16,1],[128,16,1],[128,16,1]]
        x = [sub_x.squeeze(2) for sub_x in x] # 判断第二维是否为1，若是则压缩 例：[[128,16],[128,16],[128,16]]
        x1 = torch.cat(x, 1) # 进行拼接，例：[128,48]
        x2 = self.dropout(x1) # 去除掉一些神经元防止过拟合，注意dropout之后x的维度依旧是[128,48]，并不是说我dropout的概率是0.5，去除了一半的神经元维度就变成了[128,24]，而是把x中的一些神经元的数据根据概率全部变成了0，维度依旧是[128,48]

        logits = self.fc(x2) # 全接连层 例：输入x是[128,48] 输出logits是[128,10]
        return logits, x1, x2


def train(train_iter, dev_iter, model, args):
    print("----------------- start train ---------------------------------")
    save_dir = f"{args.data_name}_cnn/{args.rate}"

    if torch.cuda.is_available():  # 判断是否有GPU，如果有把模型放在GPU上训练，速度质的飞跃
        print(f"use cuda:{torch.cuda.is_available()}")
    model=model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0)  # 梯度下降优化器，采用Adam
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in tqdm(range(1, args.epoches + 1), desc="epoches"):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            if torch.cuda.is_available():  # 如果有GPU将特征更新放在GPU上
                feature, target = feature.to(device), target.to(device)
            optimizer.zero_grad()  # 将梯度初始化为0，每个batch都是独立训练地，因为每训练一个batch都需要将梯度归零
            logits,_ ,_ = model(feature)
            loss = F.cross_entropy(logits, target)  # 计算损失函数 采用交叉熵损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 放在loss.backward()后进行参数的更新
            steps += 1
            if steps % 1 == 0:  # 每训练多少步计算一次准确率，我这边是1，可以自己修改
                corrects = (torch.max(logits, 1)[1].view(
                    target.size()).data == target.data).sum()  # logits是[128,10],torch.max(logits, 1)也就是选出第一维中概率最大的值，输出为[128,1],torch.max(logits, 1)[1]相当于把每一个样本的预测输出取出来，然后通过view(target.size())平铺成和target一样的size (128,),然后把与target中相同的求和，统计预测正确的数量
                train_acc = 100.0 * corrects / batch.batch_size  # 计算每个mini batch中的准确率
                print('steps:{} - loss: {:.6f}  acc:{:.4f}'.format(
                    steps,
                    loss.item(),
                    train_acc))

            if steps % 1 == 0:  # 每训练100步进行一次验证
                dev_acc = dev_eval(dev_iter, model)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
                    save(model, save_dir, steps)
                else:
                    if steps - last_step >= 1000:
                        print('\n提前停止于 {} steps, acc: {:.4f}%'.format(last_step, best_acc))
                        raise KeyboardInterrupt

def dev_eval(dev_iter,model):
  model.eval()
  corrects, avg_loss = 0, 0
  for batch in dev_iter:
      feature, target = batch.text, batch.label
      if torch.cuda.is_available():
          feature, target = feature.cuda(), target.cuda()
      logits,_ ,_ = model(feature)
      loss = F.cross_entropy(logits, target)
      avg_loss += loss.item()
      corrects += (torch.max(logits, 1)
                    [1].view(target.size()).data == target.data).sum()
  size = len(dev_iter.dataset)
  avg_loss /= size
  accuracy = 100.0 * corrects / size
  print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,accuracy,corrects,size))
  return accuracy

# 定义模型保存函数
def save(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = 'bestmodel.pt'
    save_bestmodel_path = os.path.join(save_dir, save_path)
    torch.save(model.state_dict(), save_bestmodel_path)

def test(test_iter,model, save_dir):
    save_path = 'bestmodel.pt'
    save_bestmodel_path = os.path.join(save_dir, save_path)

    model.load_state_dict(torch.load(save_bestmodel_path))
    if torch.cuda.is_available():  # 判断是否有GPU，如果有把模型放在GPU上训练，速度质的飞跃
        model.cuda()

    model.eval()
    corrects, avg_loss = 0, 0
    for batch in test_iter:
      feature, target = batch.text, batch.label
      if torch.cuda.is_available():
          feature, target = feature.cuda(), target.cuda()
      logits,_,_ = model(feature)
      loss = F.cross_entropy(logits, target)
      avg_loss += loss.item()
      corrects += (torch.max(logits, 1)
                    [1].view(target.size()).data == target.data).sum()
    size = len(test_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nFinal Test - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                      accuracy,
                                                                      corrects,
                                                                      size))
    return accuracy

def torchtext_file(old_file, new_file):

    datas = []
    with open(old_file, 'r', encoding='utf-8') as f:
        all = json.load(f)
        for l in all:
            datas.append((int(l["id"]), l['text'], int(l['label'])))

    with open(new_file, 'w') as fw:
        for line in datas:
            id, sent, label = line
            df = {"id":id, "text": sent, "label": label}
            encode_json = json.dumps(df)
            # 一行一行写入，并且采用print到文件的方式
            print(encode_json, file=fw)


def main(args):
    if args.data_name in ["fudan", "qinghua"]:
        print("CHA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        stopwords = stopwordslist("./stopwords/cn_stopwords.txt")
    else:
        print("ENG~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        stopwords = stopwordslist("./stopwords/en.txt")

    print("load words")

    cur_path = '{}_bert/{}/'.format(args.data_name, args.rate)

    # convert files
    nameList = ["train", "test", "val"]
    for name in nameList:
        old_files = os.path.join(cur_path, f"{name}.json")
        new_files = os.path.join(cur_path, f"new_{name}.json")
        torchtext_file(old_files, new_files)

    def cut(sentence):
        return [token for token in jieba.lcut(sentence) if token not in stopwords]
    # 声明一个Field对象，对象里面填的就是需要对文本进行哪些操作，比如这里lower=True英文大写转小写,tokenize=cut对于文本分词采用之前定义好的cut函数，sequence=True表示输入的是一个sequence类型的数据，还有其他更多操作可以参考文档
    TEXT = Field(sequential=True, lower=True, tokenize=cut)
    # 声明一个标签的LabelField对象，sequential=False表示标签不是sequence，dtype=torch.int64标签转化成整形
    LABEL = LabelField(sequential=False, use_vocab=False)

    # 这里主要是告诉torchtext需要处理哪些数据，这些数据存放在哪里，TabularDataset是一个处理scv/tsv的常用类
    train_dataset, dev_dataset, test_dataset = TabularDataset.splits(
        path=f'{args.data_name}_bert/{args.rate}/',  # 文件存放路径
        #path=f'BBC_bert/',  # 文件存放路径
        format='json',  # 文件格式
        skip_header=False,  # 是否跳过表头，我这里数据集中没有表头，所以不跳过
        train='new_train.json',
        validation='new_val.json',
        test='new_test.json',
        fields={'label': ("label", LABEL), 'text': ("text", TEXT)}  # 该版本的torchtext的不需要None
        #fields=[('id', None), ('label', LABEL), ('text', TEXT)]
        #fields={"id":None,'label':("label",LABEL), 'text': ("text",TEXT)}  # 定义数据对应的表头
    )

    #  load pretrained vector
    if not os.path.exists(vector_cache):
        os.mkdir(vector_cache)

    pretrained_name = 'crawl-300d-2M.vec'  # 预训练词向量文件名
    #pretrained_path = './drive/My Drive/TextCNN/word_embedding'  # 预训练词向量存放路径
    vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=vector_cache)
    args.vocab_size = vectors.vectors.size(0)
    args.embedding_dim = vectors.vectors.size(1) # 词向量维度

    print("-------------- build vector --------------------------")
    TEXT.build_vocab(train_dataset, dev_dataset, test_dataset,
                     vectors=vectors)
    LABEL.build_vocab(train_dataset, dev_dataset, test_dataset)
    print("-------------- finish building vector --------------------------")

    vector = vectors.vectors
    model = TextCNN(args, vector)

    print("-------------------------------- finish load model ---------------------")
    # torchtext.data.BucketIterator.splits 使用BucketIterator生成迭代器的主要是因为BucketIterator能够将样本长度接近的句子尽量放在同一个batch里面，这样假如这里我们每128个样本为一个batch，句子长度差距过大，就会给短句加入过多的无意义的<pad>，但是句子长度相近的在一个batch里面的话，就能够避免这个问题
    train_iter, dev_iter, test_iter = BucketIterator.splits(
        (train_dataset, dev_dataset, test_dataset),  # 需要生成迭代器的数据集
        batch_sizes=(128, 128, 128),  # 每个迭代器分别以多少样本为一个batch
        sort_key=lambda x: len(x.text)  # 按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
    )

    train(train_iter, dev_iter, model, args)

    save_dir = f"{args.data_name}_cnn/{args.rate}"
    test(test_iter, model, save_dir)

if __name__ =="__main__":

    args = get_params()
    main(args)




