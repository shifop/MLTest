import tensorflow as tf
import numpy as np
import json
import time
from datetime import timedelta
import os
from tqdm import tqdm


class config(object):
    corpus_size = 841660
    embedding_size = 50
    layers = 2
    epochs = 100
    batch_size = 12800
    num_sampled = 8192
    print_per_batch = 100
    cell_clip = 3
    proj_clip = 3
    lr = 1e-3
    clip_norm = 10.0
    keep = 0.9


class EMLPmodel(object):

    def __init__(self, config):
        self.config = config
        self.__createModel()

    def __perplexity(self,logits,tag):
        p = tf.nn.softmax(logits, axis=-1)
        tag_o = tf.one_hot(tag, depth=self.config.n_vocab, dtype=tf.float32)
        p = tf.reduce_sum(p * tag_o, axis=-1)
        p = tf.log(p)
        p = -tf.reduce_mean(p, axis=-1)
        perplexity = tf.exp(p)

        return tf.reduce_mean(perplexity)

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                ## input
                self.input = tf.placeholder(tf.int32, [None, None], name='input')
                self.label_f = tf.placeholder(tf.int32, [None, None], name='label_f')
                self.label_b = tf.placeholder(tf.int32, [None, None], name='label_b')
                self.content_length = tf.placeholder(tf.int32, [], name='content_length')

                self.global_step = tf.Variable(0, trainable=False)

                ## encode
                self.embedding = tf.Variable(
                    initial_value=np.random.randn(self.config.corpus_size, self.config.embedding_size)
                                  / np.sqrt( self.config.corpus_size)
                    , dtype=tf.float32, name='embedding')

                # self.embedding = tf.Variable(self.embeddinf_val, dtype=tf.float32, name='embedding')

                self.input_em_f = tf.nn.embedding_lookup(self.embedding, self.input, name='input_em_f')
                self.input_em_b = tf.nn.embedding_lookup(self.embedding, tf.reverse(self.input, [-1]),
                                                         name='input_em_b')

                self.label_f_r = tf.reshape(self.label_f, [-1, 1])
                self.label_b_r = tf.reshape(self.label_b, [-1, 1])

            with tf.device('/gpu:0'):
                ## BiLSTM
                self.forwards = []
                self.backwards = []
                for i in range(self.config.layers):
                    lstm_cell_f = tf.nn.rnn_cell.LSTMCell(
                        self.config.embedding_size * 4, num_proj=self.config.embedding_size,
                        cell_clip=self.config.cell_clip, proj_clip=self.config.proj_clip, name='forward_' + str(i))
                    lstm_cell_f = tf.nn.rnn_cell.ResidualWrapper(lstm_cell_f)
                    lstm_cell_f = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_f,
                                                                input_keep_prob=self.config.keep)
                    self.forwards.append(lstm_cell_f)

                    lstm_cell_b = tf.nn.rnn_cell.LSTMCell(
                        self.config.embedding_size * 4, num_proj=self.config.embedding_size,
                        cell_clip=self.config.cell_clip, proj_clip=self.config.proj_clip, name='backward_' + str(i))
                    lstm_cell_b = tf.nn.rnn_cell.ResidualWrapper(lstm_cell_b)
                    lstm_cell_b = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_b,
                                                                input_keep_prob=self.config.keep)
                    self.backwards.append(lstm_cell_b)

                self.outputs = []
                self.states = []

                input_cache_f = self.input_em_f
                input_cache_b = self.input_em_b
                for i in range(self.config.layers):
                    with tf.variable_scope('rnn_layers_' + str(i)):
                        output_cache_f, state_cache_f = tf.nn.dynamic_rnn(self.forwards[i], input_cache_f,
                                                                          dtype=tf.float32)
                        output_cache_b, state_cache_b = tf.nn.dynamic_rnn(self.backwards[i], input_cache_b,
                                                                          dtype=tf.float32)
                    self.outputs.append([output_cache_f, output_cache_b])
                    self.states.append([state_cache_f, state_cache_b])
                    input_cache_f = output_cache_f
                    input_cache_b = output_cache_b

                ## softmax
                # share weight
                self.softmax_w = self.embedding
                self.softmax_b = tf.zeros([self.config.corpus_size])
                self.outputs_f = tf.reshape(self.outputs[-1][0], shape=[-1, self.config.embedding_size],
                                            name='outputs_f')
                self.outputs_b = tf.reshape(self.outputs[-1][1], shape=[-1, self.config.embedding_size],
                                            name='outputs_b')

                # 计算困惑度
                logist_f = tf.matmul(self.outputs_f, self.softmax_w, transpose_b=True)
                logist_b = tf.matmul(self.outputs_b, self.softmax_w, transpose_b=True)
                self.perplexity_b = self.__perplexity(logist_b,self.label_b)
                self.perplexity_f = self.__perplexity(logist_f, self.label_f)


                self.outputs_f = tf.nn.dropout(self.outputs_f,
                                               self.config.keep)

                self.outputs_b = tf.nn.dropout(self.outputs_b,
                                               self.config.keep)

                # 完整
                self.loss_f = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, self.label_f_r, self.outputs_f
                                               , self.config.num_sampled, self.config.corpus_size),
                    name='loss_f')
                self.loss_b = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, self.label_b_r, self.outputs_b
                                               , self.config.num_sampled, self.config.corpus_size),
                    name='loss_b')

                self.loss = (tf.reduce_mean(self.loss_b) + tf.reduce_mean(self.loss_f)) / 2
                optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
                # optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.lr, momentum=0.9)

                gradients = optimizer.compute_gradients(self.loss)
                for i, (g, v) in enumerate(gradients):
                    if g is not None:
                        gradients[i] = (tf.clip_by_norm(g, self.config.clip_norm), v)  # clip gradients
                self.train_op = optimizer.apply_gradients(gradients, global_step=self.global_step)
            self.saver = tf.train.Saver()
            self.saver_var = tf.train.Saver(tf.trainable_variables())

    def __load_data(self, path):
        # 加载数据
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.loads(f.read())
        return cache

    def load_train_data(self, path=None):
        if path == None:
            path = self.config.train_data_path
        self.train_data = self.__load_data(path)

    def load_test_data(self, path=None):
        if path == None:
            path = self.config.test_data_path
        self.test_data = self.__load_data(path)

    def __batch_iter(self, paths, max_batch_size):
        for path in paths.keys():
            print('data_path is : {}'.format(path))
            batch_size = max_batch_size // paths[path]
            x = self.__load_data(path)
            length_all = len(x)
            max_index = length_all // batch_size
            if length_all % batch_size != 0:
                max_index += 1
            for index in tqdm(range(max_index)):
                input = x[index * batch_size:(index + 1) * batch_size]
                label_f = [[] for n in input]
                label_b = [[] for n in input]
                for label_index in range(len(input)):
                    label_f[label_index].extend(input[label_index][1:])
                    # 语料库最后一个字符是结束，倒数第二个是开始
                    label_f[label_index].append(self.config.corpus_size - 1)

                    label_b[label_index].append(self.config.corpus_size - 2)
                    label_b[label_index].extend(input[label_index][:-1])
                    label_b[label_index].reverse()
                yield input, label_f, label_b, len(input[0])

    def __get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def initialize(self, save_path):

        with self.graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(
                                                                    allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, save_path)

    def copy(self, save_path, load_path):

        with self.graph.as_default():
            saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(
                                                                    allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, load_path)
            saver.save(sess, save_path)

    def train(self, load_path, save_path, data_paths,dev_paths):
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            self.saver_var.restore(sess, load_path)
            iter = 0
            # 开始时间
            start_time = time.time()
            all_loss = 0
            for epoch in range(self.config.epochs):
                print('epoch : {}'.format(epoch))
                batch_iter = self.__batch_iter(data_paths, self.config.batch_size)
                for input, label_f, label_b, length in batch_iter:
                    if len(label_f) == 0:
                        continue
                    feed_dict = {self.input: input, self.label_b: label_b, self.label_f: label_f,
                                 self.content_length: length}
                    loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                    all_loss += loss
                    iter += 1
                    if iter % self.config.print_per_batch == 0:
                        perplexity_b, perplexity_f = self.evaluate(sess, dev_paths)
                        time_dif = self.__get_time_dif(start_time)
                        msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, perplexity: {2:>6.2},{3:>6.2} Time: {4}'
                        print(msg.format(iter, all_loss / self.config.print_per_batch, perplexity_b, perplexity_f, time_dif))
                        all_loss = 0
                        self.saver.save(sess, save_path)
            print('train closed')
            self.saver.save(sess, save_path)

    def evaluate(self, sess, dev_paths):
        batch_iter = self.__batch_iter(dev_paths, self.config.batch_size)
        all_b=0
        all_f=0
        count=0
        for input, label_f, label_b, length in batch_iter:
            if len(label_f) == 0:
                continue
            feed_dict = {self.input: input, self.label_b: label_b, self.label_f: label_f,
                         self.content_length: length}

            perplexity_b, perplexity_f = sess.run([self.perplexity_b, self.perplexity_f], feed_dict=feed_dict)
            count+=1
            all_b+=perplexity_b
            all_f+=perplexity_f

        return all_b/count, all_f/count