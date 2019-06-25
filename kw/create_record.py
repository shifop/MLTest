from tqdm import tqdm
import random
import tensorflow as tf
import json
from util.tools import read_json
import jieba
import jieba.posseg as pos
from jieba import analyse
import numpy as np

"""
基于词性，是否在标题中，首次出现位置，末次出现位置，词跨度
创建训练数据，由于关键词与候选词数量相差较大，采用一定策略进行样本均衡
"""

def cut(content):
    rt={}
    for x,f in pos.cut(content):
        if x not in rt:
            rt[x]=set()
        rt[x].add(f)
    return rt

def index(content):
    cache = jieba.tokenize(content)
    length = len(content)
    rt = {}
    for x in cache:
        if x[0] not in rt.keys():
            rt[x[0]]=[x[1]/length,x[1]/length,0]
        else:
            rt[x[0]][1]=x[1]/length
    for x in rt.keys():
        rt[x][2]=rt[x][1]-rt[x][0]
    return rt

if __name__ == "__main__":
    data = read_json('../data/神箭关键词提取/train_data.json')

    jieba.load_userdict('../data/神箭关键词提取/user_dicts.txt')

    v = set(['nr', 'nz', 'ns', 'n', 'eng', 'v'])
    # 生成各类特征
    tfidf = analyse.extract_tags
    d_data=[]
    for x in tqdm(data):
        tnc = x['title']+'。'+x['content']
        cache = tfidf(tnc,topK=20)
        kw = x['keywords'].split(',')
        if len(set(kw)&set(cache)) == 0:
            continue
        tag = [1 if w in kw else 0 for w in cache]
        # 基于词性，是否在标题中，首次出现位置，末次出现位置，词跨度
        ft = []
        p_map = cut(tnc)
        for w in cache:
            ft_c = [0.0,0.0]
            # 考虑词性
            if w in p_map.keys():
                if len(p_map[w]&v)!=0:
                    ft_c[0] = 1.0
            # 考虑是否在标题
            if w in x['title']:
                ft_c[1]=1.0
            # 考虑首次出现位置
            c_index = index(tnc)
            if w in c_index.keys():
                ft_c.extend(c_index[w])
            else:
                ft_c.extend([0,0,0])
            ft.append(ft_c)
        c_length = 20 -len(ft)
        c_i = len(ft)
        ft.extend([ft[c_i-1] for x in range(c_length)])
        tag.extend([tag[c_i-1] for x in range(c_length)])
        d_data.append({'ft':ft, 'tag':tag})

    # 写入record

    # 划分训练集和验证集
    random.shuffle(d_data)
    all_size = len(d_data)
    dev_data = d_data[:all_size // 2]
    t_data = d_data

    # 生成records
    train_writer = tf.python_io.TFRecordWriter('../data/神箭关键词提取/train.record')
    tag_count = []
    for x in tqdm(t_data):
        uf = []
        uuf = []
        for i, f in enumerate(x['tag']):
            if f==1:
                uf.append(i)
            else:
                uuf.append(i)

        # 生成正负样本
        for i in range(3):
            ft = [x['ft'][t] for t in uf]
            tag = [1 for t in range(len(ft))]

            for t in range(6-len(ft)):
                c = uuf[random.randint(0,len(uuf)-1)]
                ft.append(x['ft'][c])
                tag.append(0)

            features = tf.train.Features(feature={
                'ft': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(ft,np.float32).tostring()])),
                'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=tag))})
            example = tf.train.Example(features=features)
            train_writer.write(example.SerializeToString())

    train_writer.close()

    # 生成records
    train_writer = tf.python_io.TFRecordWriter('../data/神箭关键词提取/dev.record')
    for x in tqdm(dev_data):
        features = tf.train.Features(feature={
            'ft': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(x['ft'], np.float32).tostring()])),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=x['tag']))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())

    train_writer.close()


"""
vocab_size:4647 , max_length: 209
100%|██████████| 42427/42427 [01:42<00:00, 414.43it/s]
100%|██████████| 4714/4714 [00:11<00:00, 420.85it/s]
"""

