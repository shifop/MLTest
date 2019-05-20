from tqdm import tqdm
import random
import tensorflow as tf
import json

"""处理语料，生成用于分词的tfrecord
标签有：s,b,m,e
"""


def padding(data, max_size, v):
    length = len(data)
    if length < max_size:
        data.extend([v for x in range(max_size - length)])
    return data[:max_size]


if __name__ == "__main__":
    train_data = []
    cache = {'seq': [], 'tag': []}
    with open('../data/命名实体识别2/example.train', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line=='':
                train_data.append(cache)
                cache = {'seq': [], 'tag': []}
            else:
                line=line.split(' ')
                cache['seq'].append(line[0])
                cache['tag'].append(line[1])

    dev_data = []
    cache = {'seq': [], 'tag': []}
    with open('../data/命名实体识别2/example.dev', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                dev_data.append(cache)
                cache = {'seq': [], 'tag': []}
            else:
                line = line.split(' ')
                cache['seq'].append(line[0])
                cache['tag'].append(line[1])

    dev_data.append(cache)
    cache = {'seq': [], 'tag': []}
    with open('../data/命名实体识别2/example.test', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                dev_data.append(cache)
                cache = {'seq': [], 'tag': []}
            else:
                line = line.split(' ')
                cache['seq'].append(line[0])
                cache['tag'].append(line[1])

    data=[]
    data.extend(train_data)
    data.extend(dev_data)

    words = []
    for x in train_data:
        words.extend(x['seq'])

    for x in dev_data:
        words.extend(x['seq'])

    words = list(set(words))

    tag = []
    for x in train_data:
        tag.extend(x['tag'])

    for x in dev_data:
        tag.extend(x['tag'])

    tag = list(set(tag))

    # 构建tag_map,words_map
    t2i = {}
    w2i = {}
    i2t = {}
    i2w = {}
    for index, x in enumerate(tag):
        t2i[x] = index
        i2t[index] = x

    for index, x in enumerate(words):
        w2i[x] = index
        i2w[index] = x

    # 划分训练集和验证集
    random.shuffle(train_data)
    random.shuffle(dev_data)

    # 序列最大长度
    cache = [len(x['seq']) for x in data]
    max_length = max(cache)
    min_length = min(cache)
    # 词典大小
    vocab_size = len(w2i)

    print('vocab_size:%d , max_length: %d' % (vocab_size, max_length))

    # 生成records
    train_writer = tf.python_io.TFRecordWriter('../data/ner/train.record')
    for data in tqdm(train_data):
        """需要创建seq，tag，tag_p2p，mask"""
        length = len(data['seq'])
        seq = [w2i[x] for x in data['seq']]
        seq = padding(seq, max_length, vocab_size)

        tag = [t2i[x] for x in data['tag']]
        tag = padding(tag, max_length, 0)

        tag_p2p = [tag[index - 1] * len(t2i) + tag[index] for index in range(1, len(seq))]
        mask = [0 for x in range(max_length)]
        mask[length-1] = 1

        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq)),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=tag)),
            'tag_p2p': tf.train.Feature(int64_list=tf.train.Int64List(value=tag_p2p)),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=mask))})
        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())

    train_writer.close()

    dev_writer = tf.python_io.TFRecordWriter('../data/ner/dev.record')
    for data in tqdm(dev_data):
        """需要创建seq，tag，tag_p2p，mask"""
        length = len(data['seq'])
        seq = [w2i[x] for x in data['seq']]
        seq = padding(seq, max_length, vocab_size)

        tag = [t2i[x] for x in data['tag']]
        tag = padding(tag, max_length, 0)

        tag_p2p = [tag[index - 1] * len(t2i) + tag[index] for index in range(1, len(seq))]
        mask = [0 for x in range(max_length)]
        mask[length-1] = 1

        features = tf.train.Features(feature={
            'seq': tf.train.Feature(int64_list=tf.train.Int64List(value=seq)),
            'tag': tf.train.Feature(int64_list=tf.train.Int64List(value=tag)),
            'tag_p2p': tf.train.Feature(int64_list=tf.train.Int64List(value=tag_p2p)),
            'mask': tf.train.Feature(int64_list=tf.train.Int64List(value=mask))})
        example = tf.train.Example(features=features)
        dev_writer.write(example.SerializeToString())

    dev_writer.close()

    # 保存各类map
    with open('../data/ner/w2i.json','w',encoding='utf-8') as f:
        f.write(json.dumps(w2i, ensure_ascii=False))

    with open('../data/ner/i2w.json','w',encoding='utf-8') as f:
        f.write(json.dumps(i2w, ensure_ascii=False))

    with open('../data/ner/t2i.json','w',encoding='utf-8') as f:
        f.write(json.dumps(t2i, ensure_ascii=False))

    with open('../data/ner/i2t.json','w',encoding='utf-8') as f:
        f.write(json.dumps(i2t, ensure_ascii=False))


