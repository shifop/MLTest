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
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
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
        if y == 'nr':
            cache.append(['<NAME>', 1])
        elif x in dict.keys():
            cache.append([x, 1])
        else:
            cache.append([x, 0])

    # 寻找最大子串
    for c in cache:
        x = c[0]
        if c[1] == 1:
            rt.append(x)
            continue
        start_i = 0
        length = len(x)
        while length != 0:
            for index in range(len(x), start_i, -1):
                if x[start_i:index] in dict.keys():
                    rt.append(x[start_i:index])
                    start_i = index
                    length = len(x) - index
                    break
                if index == start_i + 1:
                    rt.append(x[start_i:index])
                    start_i = index
                    length = len(x) - index
    return rt


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def save_as_record(path, data, vocab_map, max_length):
    train_writer = tf.python_io.TFRecordWriter(path)
    padding_value = len(vocab_map)
    delete_count = 0
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
        if len(seq) >= max_length:
            delete_count += 1
            continue
        mask = [len(seq)]
        seq = kr.preprocessing.sequence.pad_sequences([seq], max_length, value=padding_value, padding='post',
                                                      truncating='post')
        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(seq[0].tolist(), np.int64))),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(mask, np.int64)))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())
    train_writer.close()
    print("剔除过长文本：%d" % (delete_count))


def read_txt(path='../raw_data/全国工单数据-cut.txt'):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for content in f:
            cache = content.strip()
            if len(cache.split(' ')) < 126:
                data.append(cache)
                continue
            if not cache:
                continue

            contents = []
            c = ''
            for x in cache:
                c += x
                if x in ['。', '？', '！', '······', '……']:
                    contents.append(c.strip())
                    c = ''
            if c != '':
                contents.append(c)
            data.extend(contents)
    return data


data = []
data = read_txt('./data/通用语料-cut.txt')
data.extend(read_txt('./data/0-5000000-cut.txt'))
data.extend(read_txt('./data/电信工单-语言模型训练语料100w-cut.txt'))

# 读取w2i
w2i = read_json('./data/w2i.json')

# 统计长度分布，过滤较短文本
cache = data
data = []
for x in tqdm(cache):
    length = len(x.split(' '))
    if length < 5:
        continue
    data.append(x)

# 选择可以覆盖95%数据的文本长度
select_len = 126

print('选择最大长度为：%s' % (select_len))
cache = data
data = []
for x in tqdm(cache):
    if len(x.split(' ')) > select_len:
        continue
    data.append(x)

print('文本数量：%d' % (len(data)))

# 分割训练集和验证集
length = len(data)
random.shuffle(data)
dev_data = data[:length // 100]
train_data = data[length // 100:]

random.shuffle(train_data)
random.shuffle(dev_data)

# 保存数据
save_as_record('./data/train_5g.record', train_data, w2i, select_len + 24)
save_as_record('./data/dev_5g.record', dev_data, w2i, select_len + 24)
