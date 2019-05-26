import json
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
import tensorflow.contrib.keras as kr
import jieba
import re

"""生成语言模型训练数据"""


def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.loads(f.read())
    return data

def save_as_record(path, data, vocab_map, max_length):
    train_writer = tf.python_io.TFRecordWriter(path)
    padding_value=len(vocab_map)
    start = '<start>'
    end = '<end>'
    for x in tqdm(data):
        x['label'] = id2index[x['tag']]
        cache = [vocab_map[word] for word in x['content'].split(',') if word in vocab_map.keys()]
        cache = kr.preprocessing.sequence.pad_sequences([cache], max_length, value=padding_value, padding='post', truncating='post')
        features = tf.train.Features(feature={
            'content': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(cache[0].tolist(), np.int64))),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[x['label']]))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())
    train_writer.close()


def read_txt(path='../raw_data/全国工单数据-cut.txt'):
    data=[]
    with open(path, 'r', encoding='utf-8') as f:
        while 1:
            content = f.readline().strip()
            if not content:
                break
            data.append(content)
    return data
data=read_txt()


"""读取w2i"""
w2i = read_json('../data/process/w2i.json')

# 统计长度分布，过滤较短文本
cache = data
data = []
len_map={}
for x in cache:
    length = len(x.split())
    if length not in len_map.keys():
        len_map[length]=0
    len_map[length]+=1
    if x.split(' ') < 10:
        continue
    data.append(x)

# 选择可以覆盖95%的文本长度
len_tag = [x for x in len_map.keys()]
max_len = max(len_tag)
sum_len=sum(len_tag)
count=0
select_len=0
for i in range(max_len):
    if i not in len_map.keys():
        continue
    count+=len_map[i]
    if count/sum_len>0.95:
        select_len=i
        break

print('选择最大长度为：%s'%(select_len))
cache = data
data = []
for x in cache:
    if x.split(',')>select_len:
        continue
    data.append(x)

print('文本数量：%d'%(len(data)))

# 分割训练集和验证集
length = len(data)
random.shuffle(data)
dev_data = data[:100000]
train_data = data[100000:]

random.shuffle(train_data)
random.shuffle(dev_data)

# 保存数据
save_as_record('../data/lm/train.record', train_data, w2i, select_len)
save_as_record('../data/lm/dev.record', dev_data, w2i, select_len)
