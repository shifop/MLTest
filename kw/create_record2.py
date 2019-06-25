from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait
import tensorflow as tf
from util.tools import read_json,write_json
import jieba
import jieba.posseg as pos
from jieba import analyse
import numpy as np

"""
基于词性，是否在标题中，首次出现位置，末次出现位置，词跨度
"""

def cut(content):
    rt={}
    for x,f in pos.cut(content):
        if x not in rt:
            rt[x]=set()
        rt[x].add(f)
    return rt

def content_index(content):
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


def process(params):
    data = params['data']
    name = params['name']
    jieba.load_userdict('../data/神箭关键词提取/user_dicts.txt')
    d_data = []
    v = set(['nr', 'nz', 'ns', 'n', 'eng', 'v'])
    for index,x in enumerate(data):
        tnc = x['title'] + '。' + x['content']
        tfidf = analyse.extract_tags
        cache = tfidf(tnc, topK=20)
        # 基于词性，是否在标题中，首次出现位置，末次出现位置，词跨度
        ft = []
        p_map = cut(tnc)
        for w in cache:
            ft_c = [0.0, 0.0]
            # 考虑词性
            if w in p_map.keys():
                if len(p_map[w] & v) != 0:
                    ft_c[0] = 1.0
            # 考虑是否在标题
            if w in x['title']:
                ft_c[1] = 1.0
            # 考虑首次出现位置
            c_index = content_index(tnc)
            if w in c_index.keys():
                ft_c.extend(c_index[w])
            else:
                ft_c.extend([0, 0, 0])
            ft.append(ft_c)
        c_length = 20 - len(ft)
        c_i = len(ft)
        ft.extend([ft[c_i - 1] for x in range(c_length)])
        cache.extend([cache[c_i - 1] for x in range(c_length)])
        d_data.append({'ft': ft, 'keywords': cache})
        if index%1000==0:
            print('%s:%d' % (name, index))
    return d_data


if __name__ == "__main__":
    data = read_json('../data/神箭关键词提取/docs.json')
    # 将数据分词若干份，多进程处理
    executor = ProcessPoolExecutor(6)
    tasks=[]
    for index in range(6):
        print('开启任务:%d'%(index))
        cache = data[index*20000:(index+1)*20000]
        task = executor.submit(process, {'data':cache,'name':str(index)})
        tasks.append(task)

    print('等待全部任务执行完毕')
    wait(tasks)
    d_data = []
    for x in tasks:
        print(x.done())
        d_data.extend(x.result())

    # 生成records
    train_writer = tf.python_io.TFRecordWriter('../data/神箭关键词提取/test2.record')
    for x in tqdm(d_data):
        print(np.array(x['ft']).shape)
        features = tf.train.Features(feature={
            'ft': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(x['ft'],np.float32).tostring()]))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())

    train_writer.close()

    write_json(d_data,'../data/神箭关键词提取/test2.json')


"""
vocab_size:4647 , max_length: 209
100%|██████████| 42427/42427 [01:42<00:00, 414.43it/s]
100%|██████████| 4714/4714 [00:11<00:00, 420.85it/s]
"""

