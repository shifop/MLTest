from tqdm import tqdm
import random
import tensorflow as tf
import json

"""处理语料，生成tfrecord
"""


def padding(data, max_size, v):
    length = len(data)
    if length < max_size:
        data.extend([v for x in range(max_size - length)])
    return data[:max_size]


if __name__ == "__main__":
    data = []
    with open('../data/词性标注@人民日报199801.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())

    cut_data = []
    for x in data:
        cut_data.append(x.split(' '))

    tag = []
    train_data = []
    for x in cut_data:
        # 对文本再次切割,以句号，感叹号，问好，；
        words = []
        tags = []
        for y in x:
            cache = y.split('/')
            if len(cache) < 2:
                continue
            else:
                tag.append('/' + y.split('/')[1].split(']')[0])
                words.append(cache[0])
                tags.append(tag[-1])
        if len(words) != 0:
            start = 0
            for index in range(len(words)):
                if words[index]== '……' or words[index] == '。' or words[index] == '；' or words[index] == '？' or words[index] == '！' or index == len(words)-1:
                    if index+1-start < 100 and index+1-start > 1:
                        train_data.append({'seq': words[start:index+1], 'tag': tags[start:index+1]})
                    start = index+1

    tag = list(set(tag))

    words = []
    for x in train_data:
        words.extend(x['seq'])

    words = list(set(words))

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
    all_size = len(train_data)
    dev_data = train_data[:all_size // 10]
    t_data = train_data[all_size // 10:]

    # 序列最大长度
    cache = [len(x['seq']) for x in train_data]
    max_length = max(cache)
    min_length = min(cache)
    # 词典大小
    vocab_size = len(w2i)

    print('vocab_size:%d , max_length: %d' % (vocab_size, max_length))

    # 生成records
    train_writer = tf.python_io.TFRecordWriter('../data/train.record')
    for data in tqdm(t_data):
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

    dev_writer = tf.python_io.TFRecordWriter('../data/dev.record')
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
    with open('../data/w2i.json','w',encoding='utf-8') as f:
        f.write(json.dumps(w2i, ensure_ascii=False))

    with open('../data/i2w.json','w',encoding='utf-8') as f:
        f.write(json.dumps(i2w, ensure_ascii=False))

    with open('../data/t2i.json','w',encoding='utf-8') as f:
        f.write(json.dumps(t2i, ensure_ascii=False))

    with open('../data/i2t.json','w',encoding='utf-8') as f:
        f.write(json.dumps(i2t, ensure_ascii=False))


