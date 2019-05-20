# coding: utf-8
"""bilstm-crf 用于词性标注"""

from datetime import timedelta

import numpy as np
import tensorflow as tf
import time
import math
from tqdm import tqdm

def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([577], tf.int64),
                                           'tag': tf.FixedLenFeature([577], tf.int64),
                                           'tag_p2p': tf.FixedLenFeature([576], tf.int64),
                                           'mask': tf.FixedLenFeature([577], tf.int64)
                                       }
                                       )
    return features['seq'], features['tag'], features['tag_p2p'], features['mask']


def parser_dev(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([577], tf.int64)
                                       }
                                       )
    return features['seq']


class TCNNConfig(object):
    """CNN配置参数"""

    seq_length = 577
    embedding_size = 50
    vocab_size = 4465
    pos_size = 44
    batch_size = 256
    learning_rate = 1e-3

    print_per_batch = 20  # 每多少轮输出一次结果
    dev_per_batch = 500  # 多少轮验证一次

    train_data_path = '../data/train.record'
    train_data_size = 64
    test_data_path = '../data/dev.record'
    dev_data_path = '../data/dev.record'
    num_epochs = 200


class TextCNN(object):

    def __init__(self, config):
        self.config = config
        self.__createModel()
        self.log_writer = tf.summary.FileWriter('../log/20190511214400',self.graph)
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

    def __crf_loss(self, seq, tag, tag_p2p, h_v, mask, config):
        """
        计算crf版的loss
        :param seq: 待标注序列
        :param tag: 词性标注 ,shape为[batch, seq_length-1]
        :param h_v: 词-词性得分矩阵
        :param config: 相关配置信息
        :return:
        """
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                mask = tf.argmax(mask, axis=-1)
                seq_mark = tf.sequence_mask(mask,config.seq_length,dtype=tf.float32)
                g_v = tf.get_variable('g_v', [config.pos_size*config.pos_size,1], tf.float32)
                # g_v = tf.reshape(g_v, [config.pos_size, config.pos_size])
                # loss,_=tf.contrib.crf.crf_log_likelihood(h_v,tag,mask,g_v)


                # 计算当前标注序列的得分
                g_c = tf.nn.embedding_lookup(g_v, tag_p2p)
                g_c = tf.reshape(g_c, [-1, config.seq_length-1])
                g_c = g_c * seq_mark[:,1:]
                g_c = tf.reduce_sum(g_c, axis=-1, keepdims=True)

                tag_oh = tf.one_hot(tag, config.pos_size)
                h_c = tf.reduce_sum(tag_oh*h_v, axis=-1)
                h_c = h_c * seq_mark
                h_c = tf.reduce_sum(h_c, axis=-1, keepdims=True)

                # 计算归一化因子Z
                g_v = tf.reshape(g_v, [config.pos_size, config.pos_size])
                g_v = tf.expand_dims(g_v, axis=0)
                Z = []
                Z.append(tf.expand_dims(h_v[:, 0, :], axis=2))
                for index in range(1, config.seq_length):
                    cache = Z[index-1]
                    cache = cache+g_v
                    cache = h_v[:,index,:]+tf.reduce_logsumexp(cache, axis=1)
                    Z.append(tf.expand_dims(cache,axis=2))

                Z = tf.concat(Z, axis=2)
                Z = tf.reduce_logsumexp(Z, axis=1)
                mask_1 = tf.maximum(
                    tf.constant(0, dtype=mask.dtype),
                    mask - 1)
                Z = Z*tf.one_hot(mask_1,config.seq_length)
                Z = tf.reduce_sum(Z,axis=-1)
                loss = Z - g_c - h_c

        return tf.reduce_mean(loss)

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


    def __get_data(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(self.config.batch_size)
        if is_train:
            dataset = dataset.shuffle(64 * 10)
            dataset = dataset.prefetch(64)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        seq, tag, tag_p2p, mask= iter.get_next()

        seq = tf.cast(seq, tf.int32)
        tag = tf.cast(tag, tf.int32)
        tag_p2p = tf.cast(tag_p2p, tf.int32)
        mask = tf.cast(mask, tf.float32)

        seq = tf.reshape(seq, [-1, self.config.seq_length])
        tag = tf.reshape(tag, [-1, self.config.seq_length])
        tag_p2p = tf.reshape(tag_p2p, [-1, self.config.seq_length - 1])
        mask = tf.reshape(mask, [-1, self.config.seq_length])

        # mask2 = tf.

        return seq, tag, tag_p2p, mask, iter.make_initializer(dataset)

    def __get_dev_data(self, path, parser):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        seq = iter.get_next()
        # 担心数据不一致，所以转化一次
        seq = tf.reshape(seq, [-1, self.config.seq_length])

        return seq, iter.make_initializer(dataset)

    def __train(self, seq, tag, tag_p2p, mask):
        h_v = self.__encode(seq, self.config)
        loss = self.__crf_loss(seq, tag, tag_p2p, h_v, mask, self.config)
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                # tag-tag 得分矩阵
                g_v = tf.get_variable('g_v', [config.pos_size*config.pos_size,1], tf.float32)
                g_v = tf.reshape(g_v, [config.pos_size, config.pos_size])
        return loss,h_v,g_v

    def __dev(self, inputX, input_index, inputY=None, is_test=False):
        input = tf.split(inputX, 12)
        outputs = self.encode_dev(input[-1], input_index, self.config)

        if is_test:
            return outputs

        else:
            with tf.name_scope("loss"):
                loss = tf.sqrt(tf.reduce_mean(tf.pow(inputY - outputs, 2)))
            return loss

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            seq, tag, tag_p2p, self.mask, self.train_data_op=self.__get_data(self.config.train_data_path, parser, is_train=True)
            dev_seq, dev_tag, dev_tag_p2p, self.dev_mask, self.dev_data_op = self.__get_data(self.config.test_data_path, parser)
            test_seq, self.p_data_op = self.__get_dev_data(self.config.dev_data_path, parser_dev)

            self.loss,self.h_v,self.g_v = self.__train(seq, tag, tag_p2p, self.mask)
            self.dev_loss,self.dev_h_v,self.dev_g_v = self.__train(dev_seq, dev_tag, dev_tag_p2p, self.dev_mask)

            self.summary_train_loss = tf.summary.scalar( 'train_loss', self.loss)
            self.summary_dev_loss = tf.summary.scalar('dev_loss', self.dev_loss)

            self.dev_tag = dev_tag

            # 优化器
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate,momentum=0.9)
            with tf.control_dependencies(update_ops):
                self.optim = optimizer.minimize(self.loss,global_step=tf.train.get_global_step())

            self.saver = tf.train.Saver()
            self.saver_v = tf.train.Saver(tf.trainable_variables())

            self.merged = tf.summary.merge_all()

    def train(self, load_path, save_path):
        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 9000  # 如果超过指定轮未提升，提前结束训练
        all_loss = 0

        flag = False
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            sess.run(tf.global_variables_initializer())
            # self.saver.restore(sess, load_path)

            for epoch in range(self.config.num_epochs):
                if flag:
                    break
                print('Epoch:', epoch + 1)
                sess.run(self.train_data_op)
                for step in tqdm(range(20864//self.config.batch_size+1)):
                    if total_batch % self.config.print_per_batch == 0:
                        if total_batch % self.config.dev_per_batch == 0 and total_batch!=0:
                            # 跑验证集
                            dev_loss, dev_acc = self.evaluate(sess,total_batch//self.config.dev_per_batch-1)
                            if min_loss == -1 or min_loss <= dev_acc:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_acc
                            else:
                                improved_str = ''

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Val loss: {2:>6.5}, Val acc:{3:>6.3} Time: {4} {5}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch, dev_loss, dev_acc,
                                             time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Time: {2}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch, time_dif))
                        all_loss = 0

                    loss_train, summary, _ = sess.run([self.loss,self.summary_train_loss, self.optim])  # 运行优化
                    self.log_writer.add_summary(summary,total_batch)
                    all_loss += loss_train
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:
                    break

    def evaluate(self, sess, count):
        sess.run(self.dev_data_op)
        all_loss = 0
        dev_count = 0
        data_size = 6955//self.config.batch_size+1
        for step in range(data_size):
            loss_train, dev_h_v, dev_g_v, tag, mask, summary = sess.run([self.dev_loss,self.dev_h_v,self.dev_g_v, self.dev_tag,
                                                          self.dev_mask, self.summary_dev_loss])
            # 计算预测值
            # p_tag=[tf.contrib.crf.viterbi_decode(dev_h_v[index,:,:], dev_g_v)[0] for index in range(dev_h_v.shape[0])]
            mask = mask.argmax(axis=1)
            p_tag = self.__p(dev_h_v,dev_g_v,mask)
            # 计算准确率
            p_tag = np.array(p_tag)
            size=0
            acc_v=0
            for index,x in enumerate(mask):
                size+=x
                cache=np.array(p_tag[index])==tag[index,:x]
                acc_v+=cache.astype(int).sum()

            if not math.isnan(loss_train):
                self.log_writer.add_summary(summary, count*data_size + step)
                all_loss += loss_train
                dev_count += 1
        return all_loss/dev_count, acc_v/size

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config = TCNNConfig()
    oj = TextCNN(config)
    path = '../model/20190511214400/model.ckpt'
    oj.train(path, path)