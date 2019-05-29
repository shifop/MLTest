import json
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
import tensorflow.contrib.keras as kr
import jieba.posseg as pseg
import re

"""生成语言模型训练数据"""


def is_alphabet(uchar):
    if (u'\u0041' <= uchar<=u'\u005a') or (u'\u0061' <= uchar<=u'\u007a'):
        return True
    else:
        return False


def bpe(word, dict):
    # 处理字母和数字
    if word.isdigit():
        if ('<%dNUM>' % (len(word))) in dict.keys():
            return [('<%dNUM>' % (len(word)))]
        return ['<NUM>']
    if is_alphabet(word):
        if ('<%dCHAR>' % (len(word))) in dict.keys():
            return [('<%dCHAR>' % (len(word)))]
        return ['<CHAR>']

    # 再次分词
    rt = []
    word = [[x, y] for x, y in pseg.cut(word)]
    cache = []
    for x, y in word:
        if y == 'x':
            continue
        if x in dict.keys():
            rt.append(x)
        else:
            cache.append(x)

    # 寻找最大子串
    for x in cache:
        start_i = 0
        length = len(x)
        while length == 0:
            for index in range(len(x), start_i, -1):
                if x[start_i:index] in dict.keys():
                    rt.append(x[start_i:index])
                    start_i = index
                    length = len(x) - index
                    break
                if index == start_i + 1:
                    start_i = index
                    length = len(x) - index
    return rt

def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.loads(f.read())
    return data

def save_as_record(path, data, vocab_map, max_length):
    train_writer = tf.python_io.TFRecordWriter(path)
    padding_value=len(vocab_map)
    start = '<START>'
    end = '<END>'
    for x in tqdm(data):
        # 转化为id
        cache = []

        for word in x.split(' '):
            if word == ' ' or word == '':
                continue
            if word in vocab_map.keys():
                cache.append(vocab_map[word])
            else:
                word = bpe(word, vocab_map)
                cache.extend([vocab_map[w] for w in word if w in vocab_map.keys()])

        seq = [vocab_map['<START>']]
        seq.extend(cache)
        seq.append(vocab_map['<END>'])
        mask = [len(seq)]
        seq = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=padding_value, padding='post', truncating='post')
        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(mask, np.int64)))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())
    train_writer.close()


def read_txt(path='../raw_data/全国工单数据-cut.txt'):
    data=[]
    with open(path, 'r', encoding='utf-8') as f:
        for content in f:
            content = content.strip()
            if not content:
                continue
            data.append(content)
    return data


def read_txt2(path='../raw_data/全国工单数据-cut.txt'):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for content in f:
            cache = content.strip()
            if not cache:
                continue

            contents = []
            c = ''
            for x in cache:
                c += x
                if x in ['。','？','！','······','……']:
                    contents.append(c.strip())
                    c = ''
            if c != '':
                contents.append(c)

            for content in contents:
                length = len(content.split(' '))
                if length not in data:
                    data[length] = 0
                data[length] += 1
    return data


len_map={}

cache = read_txt2('../data/lm/通用语料-cut.txt')
for key in cache.keys():
    if key not in len_map.keys():
        if key < 5:
            continue
        len_map[key]=0
    len_map[key] += cache[key]


"""
cache = read_txt2('../data/lm/0-5000000-cut.txt')
for key in cache.keys():
    if key not in len_map.keys():
        if key < 5:
            continue
        len_map[key]=0
    len_map[key] += cache[key]


cache = read_txt2('../data/lm/电信工单-语言模型训练语料100w-cut.txt')
for key in cache.keys():
    if key not in len_map.keys():
        if key < 5:
            continue
        len_map[key]=0
    len_map[key] += cache[key]
"""

# 选择可以覆盖95%数据的文本长度
len_tag = [x for x in len_map.keys()]
len_list = [len_map[x] for x in len_tag]
max_len = max(len_tag)
sum_len=sum(len_list)
count=0
select_len=0
print('数据最大长度：%d' % (max_len))
for i in range(max_len):
    if i not in len_map.keys():
        continue
    count += len_map[i]
    if count/sum_len > 0.95:
        select_len = i
        break

print('选择最大长度为：%s'%(select_len))
print('数据总量:%d'%(sum_len))

# 数据最大长度：173562
# 选择最大长度为：390
# 数据总量:33263305

"""
69
58
124
"""