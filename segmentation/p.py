# coding: utf-8
"""bilstm-crf 用于词性标注"""

from datetime import timedelta

import numpy as np
import tensorflow as tf
import json


class TCNNConfig(object):
    """CNN配置参数"""

    seq_length = 209
    embedding_size = 50
    vocab_size = 4648
    pos_size = 4
    batch_size = 256
    learning_rate = 1e-3

    print_per_batch = 20  # 每多少轮输出一次结果
    dev_per_batch = 500  # 多少轮验证一次

    train_data_path = '../data/seg/train.record'
    train_data_size = 64
    test_data_path = '../data/seg/dev.record'
    dev_data_path = '../data/seg/dev.record'
    num_epochs = 200

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


class TextCNN(object):

    def __init__(self, config):
        self.config = config
        self.__createModel()
        self.train_data = {}
        self.test_data = {}

    def initialize(self, save_path):
        with self.graph.as_default():
            saver = tf.train.Saver(name='save_saver')
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, save_path)

    def __encode(self, seq, config):
        """
        使用bilstm编码
        :param seq: 待标注序列
        :param config: 相关配置信息
        :return: h(y_i|X)
        """
        with tf.name_scope("encode"):
            with tf.variable_scope("var-encode", reuse=tf.AUTO_REUSE):
                embedding = tf.get_variable('embedding', [config.vocab_size, config.embedding_size], tf.float32)
                seq_em = tf.nn.embedding_lookup(embedding, seq)
                # 创建两个lstm_cell
                f_cell = tf.nn.rnn_cell.LSTMCell(config.embedding_size, name='f_cell', reuse=tf.AUTO_REUSE)
                b_cell = tf.nn.rnn_cell.LSTMCell(config.embedding_size, name='b_cell', reuse=tf.AUTO_REUSE)

                output, _= tf.nn.bidirectional_dynamic_rnn(f_cell, b_cell, seq_em, dtype=tf.float32)
                output = tf.concat(output, axis=-1)

                h_v = tf.layers.dense(output, config.pos_size, name='dense', reuse=tf.AUTO_REUSE)

        return h_v

    def __p(self, h_v, g_v, mask):
        """
        使用动态规划找到最大可能性的标注
        :param h_v: 词-词性得分矩阵 [batch, seq_length, pos_size]
        :param g_v: 词性-词性转移得分矩阵 [pos_size, pos_size]
        :param mask: 有效长度
        :return:
        """
        rt=[]
        for index,h in enumerate(h_v):
            pos_size = g_v.shape[0]
            seq_length = mask[index]
            path = [[] for x in range(pos_size)]  # 在当前阶段，标注为不同词性的最大得分的序列
            score = np.zeros([seq_length, pos_size], np.float32)  # 在当前阶段，标注为对应词性的最大得分

            score[0, :] += h[0, :]
            for i in range(pos_size):
                path[i].append(i)
            for index in range(1, seq_length):
                # 计算在当前阶段，标注为不同词性的最大得分
                path_cache = path
                path = [[] for x in range(pos_size)]
                for i in range(pos_size):
                    cache = np.array([score[index - 1, y] + g_v[y, i] + h[index, i] for y in range(pos_size)])
                    max_v = cache.max(axis=0)
                    max_index = cache.argmax(axis=0)
                    # 更新得分
                    score[index, i] = max_v
                    # 更新路径
                    path[i].extend(path_cache[max_index])
                    path[i].append(i)

            max_index = score[seq_length - 1, :].argmax(axis=-1)
            rt.append(path[max_index])
        return rt

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.seq = tf.placeholder(tf.int32, shape=(1, 209), name='seq')
            self.h_v = self.__encode(self.seq, self.config)
            with tf.name_scope("decode"):
                with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                    # tag-tag 得分矩阵
                    g_v = tf.get_variable('g_v', [config.pos_size * config.pos_size, 1], tf.float32)
                    self.g_v = tf.reshape(g_v, [config.pos_size, config.pos_size])

            self.saver_v = tf.train.Saver(tf.trainable_variables())


    def evaluate(self, sess, content, mask):
        h_v, g_v = sess.run([self.h_v,self.g_v], feed_dict={self.seq:content})
        # 计算预测值
        p_tag = self.__p(h_v,g_v,mask)
        return p_tag[0][:mask[0]]



if __name__=='__main__':
    config = TCNNConfig()
    oj = TextCNN(config)
    w2i = read_json('../data/seg/w2i.json')
    path = '../model/20190511214400/model.ckpt'
    with tf.Session(graph=oj.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        oj.saver_v.restore(sess, path)
        seq=[x for x in '实现祖国的完全统一，是海内外全体中国人的共同心愿。']
        content = [w2i[x] for x in seq]
        mask=[len(content)]
        content.extend([4647 for x in range(209-len(content))])
        rt = oj.evaluate(sess,[content],mask)
        p=[]
        tag = ['s','b','m','e']
        for index,x in enumerate(seq):
            p.append(x+'/'+tag[rt[index]])
        print(' '.join(p))
