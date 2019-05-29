# coding: utf-8
"""bilstm-crf 用于词性标注"""

from datetime import timedelta
from language_model.gpt_2 import *
import numpy as np
import tensorflow as tf
import time
import math
from tqdm import tqdm

def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([420], tf.int64),
                                           'tag': tf.FixedLenFeature([420], tf.int64),
                                           'mask': tf.FixedLenFeature([1], tf.int64)
                                       }
                                       )
    return features['seq'], features['tag'], features['mask']


def parser_dev(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([420], tf.int64)
                                       }
                                       )
    return features['seq']


class Config(object):
    """CNN配置参数"""
    n_vocab = 100000  # 词库大小
    n_ctx = 140  # 序列最大长度
    n_embd = 768  # 词向量维度
    n_head = 12  # 注意力头数
    n_layer = 12  # 网络层数
    num_sampled = 8192

    batch_size = 256
    learning_rate = 0.00002

    print_per_batch = 20  # 每多少轮输出一次结果
    dev_per_batch = 500  # 多少轮验证一次

    train_data_path = '../data/train.record'
    train_data_size = 64
    test_data_path = '../data/dev.record'
    dev_data_path = '../data/dev.record'
    num_epochs = 200


class GPT_2(object):

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

    def __get_dev_data(self, path, parser):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        seq = iter.get_next()
        # 担心数据不一致，所以转化一次
        seq = tf.reshape(seq, [-1, self.config.n_ctx])

        return seq, iter.make_initializer(dataset)

    def __get_data(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(self.config.batch_size)
        if is_train:
            dataset = dataset.shuffle(64 * 10)
            dataset = dataset.prefetch(64)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        seq, tag, mask= iter.get_next()

        seq = tf.cast(seq, tf.int32)
        tag = tf.cast(tag, tf.int32)
        mask = tf.cast(mask, tf.int32)

        seq = tf.reshape(seq, [-1, self.config.n_ctx])
        tag = tf.reshape(tag, [-1, self.config.n_ctx])
        tag = tf.concat([tag[:, 1:], tag[:, -1:]], axis=-1)
        mask = tf.reshape(mask, [-1])
        mask = tf.sequence_mask(mask, self.config.n_ctx)

        # 创建tag

        return seq, tag, mask, iter.make_initializer(dataset)

    def __train(self, seq, tag, mask):
        """
        计算损失函数
        :param seq: 序列
        :param tag: 预测序列
        :param mask: 掩码，去除无效部分
        :return:
        """
        result = model(self.config, seq, None, "gpt", tf.AUTO_REUSE)
        h_flat = result['h_flat']
        # 调整标注序列
        h_flat = tf.reshape(h_flat,[-1, self.config.n_embd])
        wte = result['wte']
        basic = tf.zeros([self.config.n_vocab])
        tag = tf.reshape(tag, [-1,1])
        loss = tf.nn.sampled_softmax_loss(wte, basic, tag, h_flat, self.config.num_sampled, self.config.n_vocab)
        loss = tf.reshape(loss,[-1, self.config.n_ctx])
        loss = tf.boolean_mask(loss, mask)

        return tf.reduce_mean(loss)

    def __dev(self,seq, tag, mask):
        """
        计算困惑度
        :param seq:
        :param tag:
        :param mask:
        :return:
        """
        result = model(self.config, seq, None, "gpt", tf.AUTO_REUSE)
        h_flat = result['h_flat']
        wte = result['wte']
        h_flat = tf.reshape(h_flat, [-1, self.config.n_embd])
        # 标签的wte
        tag_w = tf.reshape(tf.gather(wte, tag), [-1, self.config.n_embd])
        logits_tag = tf.exp(tf.reduce_sum(h_flat * tag_w, axis=-1))
        # 循环计算logits
        logits_all = tf.reduce_sum(tf.exp(tf.matmul(h_flat, wte[0:50000], transpose_b=True)), axis=-1)
        def cond(logits, start_i, end_i, max_len):
            return end_i < max_len
        def body(logits, start_i, end_i, max_len):
            start_i += 50000
            end_i += 50000
            logits = logits+ tf.reduce_sum(tf.exp(tf.matmul(h_flat, wte[start_i:end_i], transpose_b=True)), axis=-1)
            return logits,start_i,end_i,max_len
        logits_all, _, _, _ = tf.while_loop(cond, body, [logits_all, 0, 50000, config.n_vocab])
        # logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = logits_tag/logits_all
        logits = tf.reshape(logits, [-1, self.config.n_ctx])
        # p = tf.nn.softmax(logits, axis=-1)
        p = logits
        # tag_o = tf.one_hot(tag, depth=self.config.n_vocab, dtype=tf.float32)
        # p = tf.reduce_sum(p*tag_o, axis=-1)
        p = tf.boolean_mask(tf.log(p), mask)
        p = -tf.reduce_mean(p, axis=-1)
        perplexity = tf.exp(p)

        return tf.reduce_mean(perplexity)


    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            seq, tag, self.mask, self.train_data_op=self.__get_data(self.config.train_data_path, parser, is_train=True)
            dev_seq, dev_tag, self.dev_mask, self.dev_data_op = self.__get_data(self.config.test_data_path, parser)
            test_seq, self.p_data_op = self.__get_dev_data(self.config.dev_data_path, parser_dev)

            self.loss = self.__train(seq, tag, self.mask)
            self.perplexity = self.__dev(dev_seq, dev_tag, self.dev_mask)

            self.summary_train_loss = tf.summary.scalar( 'train_loss', self.loss)
            self.summary_dev_loss = tf.summary.scalar('perplexity', self.perplexity)

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
                for step in tqdm(range(1078209//self.config.batch_size+1)):
                    if total_batch % self.config.print_per_batch == 0:
                        if total_batch % self.config.dev_per_batch == 0 and total_batch!=0:
                            # 跑验证集
                            dev_loss = self.evaluate(sess,total_batch//self.config.dev_per_batch-1)
                            if min_loss == -1 or min_loss <= dev_loss:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_loss
                            else:
                                improved_str = ''

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Val loss: {2:>6.5}, Time: {3} {4}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch, dev_loss,
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
        data_size = 4714//self.config.batch_size+1
        for step in range(data_size):
            perplexity, summary = sess.run([self.perplexity, self.summary_dev_loss])

            if not math.isnan(perplexity):
                self.log_writer.add_summary(summary, count*data_size + step)
                all_loss += perplexity
                dev_count += 1
        return all_loss/dev_count

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

if __name__=='__main__':
    config = Config()
    oj = GPT_2(config)
    path = '../model/20190511214400/model.ckpt'
    oj.train(path, path)