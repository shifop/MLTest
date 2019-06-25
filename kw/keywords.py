"""
提取关键词
基于词性，是否在标题中，首次出现位置，末次出现位置，词跨度
"""

from datetime import timedelta

import numpy as np
import tensorflow as tf
import time
import math
from tqdm import tqdm


def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'ft': tf.FixedLenFeature([], tf.string),
                                           'tag': tf.FixedLenFeature([6], tf.int64)
                                       }
                                       )
    return features['ft'], features['tag']


def parser2(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'ft': tf.FixedLenFeature([], tf.string),
                                           'tag': tf.FixedLenFeature([20], tf.int64)
                                       }
                                       )
    return features['ft'], features['tag']


def parser_dev(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'ft': tf.FixedLenFeature([], tf.string)
                                       }
                                       )
    return features['ft']


def get_data_count(path):
    c = 0
    for record in tf.python_io.tf_record_iterator(path):
        c += 1
    return c


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

    def __get_data(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(self.config.batch_size)
        if is_train:
            dataset = dataset.shuffle(64 * 10)
            dataset = dataset.prefetch(64)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        ft, tag= iter.get_next()

        ft = tf.decode_raw(ft, tf.float32)
        tag = tf.cast(tag, tf.float32)

        ft = tf.reshape(ft, [-1, 6, 5])
        tag = tf.reshape(tag, [-1, 6])

        # mask2 = tf.
        return ft, tag, iter.make_initializer(dataset)

    def __get_data2(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(self.config.batch_size)
        if is_train:
            dataset = dataset.shuffle(64 * 10)
            dataset = dataset.prefetch(64)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        ft, tag= iter.get_next()

        ft = tf.decode_raw(ft, tf.float32)
        tag = tf.cast(tag, tf.float32)

        ft = tf.reshape(ft, [-1, 20, 5])
        tag = tf.reshape(tag, [-1, 20])

        # mask2 = tf.
        return ft, tag, iter.make_initializer(dataset)

    def __get_dev_data(self, path, parser, is_train=False):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        dataset = dataset.batch(self.config.batch_size)
        if is_train:
            dataset = dataset.shuffle(64 * 10)
            dataset = dataset.prefetch(64)
        iter = tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
        ft = iter.get_next()

        ft = tf.decode_raw(ft, tf.float32)

        ft = tf.reshape(ft, [-1, 20, 5])

        return ft, iter.make_initializer(dataset)

    def __train(self, ft, tag, size=6, training=False):
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                p = tf.layers.dense(ft, 20, name='dense', reuse=tf.AUTO_REUSE)
                p = tf.layers.dropout(p, 0.5,training=training)
                p = tf.layers.dense(p, 1, name='dense1', reuse=tf.AUTO_REUSE)
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=p, labels=tf.reshape(tag, [-1, size, 1]))
                pt = tf.reshape(tf.sigmoid(p), [-1, size])
        return tf.reduce_mean(loss), tag, pt


    def __p(self, ft):
        with tf.name_scope("decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                p = tf.layers.dense(ft, 20, name='dense', reuse=tf.AUTO_REUSE)
                p = tf.layers.dropout(p, 0.5, training=False)
                p = tf.layers.dense(p, 1, name='dense1', reuse=tf.AUTO_REUSE)
                pt = tf.sigmoid(p)
        return pt

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            ft, tag, self.train_data_op=self.__get_data(self.config.train_data_path, parser, is_train=True)
            dev_ft, dev_tag, self.dev_data_op = self.__get_data2(self.config.dev_data_path, parser2)
            test_ft, self.test_data_op = self.__get_dev_data(self.config.test_data_path, parser_dev)

            self.test_pt = self.__p(test_ft)
            self.loss,self.acc,self.pt = self.__train(ft, tag, 6, True)
            self.dev_loss,self.dev_acc,self.dev_pt = self.__train(dev_ft, dev_tag, 20, False)

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
        all_loss = 0
        all_acc = 0
        train_data_count = get_data_count(self.config.train_data_path)
        dev_data_count = get_data_count(self.config.dev_data_path)
        require_improvement = (train_data_count//self.config.batch_size) * 300  # 如果超过指定轮未提升，提前结束训练

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
                for step in tqdm(range(train_data_count//self.config.batch_size+1)):
                    if total_batch % self.config.print_per_batch == 0 and total_batch>(train_data_count//self.config.batch_size+1)*20:
                        if total_batch % self.config.dev_per_batch == 0:
                            # 跑验证集
                            if total_batch%20==0:
                                print('')
                            dev_loss, dev_acc = self.evaluate(sess,dev_data_count)
                            if min_loss == -1 or min_loss <= dev_acc:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_acc
                            else:
                                improved_str = ''
                                self.saver.save(sess=sess, save_path=save_path)

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train acc: {2:>6.2} ' \
                                  'Val loss: {3:>6.5}, Val acc:{4:>6.3} Time: {5} {6}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch,
                                             all_acc / self.config.print_per_batch,
                                             dev_loss, dev_acc,
                                             time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train acc: {2:>6.2} Time: {3}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch,
                                             all_acc / self.config.print_per_batch, time_dif))
                        all_loss = 0
                        all_acc = 0

                    loss_train, acc_train, pt_train, summary, _ = sess.run([self.loss, self.acc, self.pt, self.summary_train_loss, self.optim])  # 运行优化
                    acc_train = self.__acc(acc_train, pt_train)
                    self.log_writer.add_summary(summary, total_batch)
                    all_loss += loss_train
                    all_acc += acc_train
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
        all_acc = 0
        dev_count = 0
        data_size = count//self.config.batch_size+1
        for step in range(data_size):
            loss_train, dev_acc, dev_pt, summary = sess.run([self.dev_loss,self.dev_acc,self.dev_pt, self.summary_dev_loss])
            dev_acc = self.__acc(dev_acc, dev_pt,[0.8,0.2])
            all_loss += loss_train
            all_acc += dev_acc
            dev_count+=1
        return all_loss/dev_count, all_acc/dev_count

    def __acc(self, tag, p, rate=[0.5,0.5]):
        p = p > 0.5
        c_tag = tag > 0.5
        tag = np.reshape(tag,[-1])

        acc1 = np.reshape(c_tag==p,[-1])*tag
        acc1 = sum(acc1)/sum(tag)
        acc2 = np.reshape(c_tag == p, [-1]) * (1-tag)
        acc2 = sum(acc2)/sum(1-tag)
        acc = (acc1*rate[0]+acc2*rate[1])
        return acc

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))


class TCNNConfig(object):
    """CNN配置参数"""

    pos_size = 4
    batch_size = 256
    learning_rate = 1e-3

    print_per_batch = 2  # 每多少轮输出一次结果
    dev_per_batch = 2  # 多少轮验证一次

    train_data_path = '../data/神箭关键词提取/train.record'
    train_data_size = 64
    test_data_path = '../data/神箭关键词提取/dev.record'
    dev_data_path = '../data/神箭关键词提取/dev.record'
    num_epochs = 50

if __name__=='__main__':
    config = TCNNConfig()
    oj = TextCNN(config)
    path = '../model/20190602202700/model.ckpt'
    oj.train(path, path)