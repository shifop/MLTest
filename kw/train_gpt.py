# coding: utf-8

from datetime import timedelta
import numpy as np
import tensorflow as tf
import time
import math
from tqdm import tqdm
from tensorflow.contrib.training import HParams

"""
训练标题生成模型，使用标题生成模型中的中间输出预测关键词
"""


def get_data_count(path):
    c = 0
    for record in tf.python_io.tf_record_iterator(path):
        c += 1
    return c


def default_hparams():
    return HParams(
        n_vocab=100000,  # 词库大小
        n_ctx=1024,  # 序列最大长度
        n_embd=768,  # 词向量维度
        n_head=12,  # 注意力头数
        n_layer=12,  # 网络层数
    )


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        with tf.device('/cpu:0'):
            g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
            b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    """
    全连接层
    :param x: 输入
    :param scope: 变量域名
    :param nf: 输出维度
    :param w_init_stdev:
    :return:
    """
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        with tf.device('/cpu:0'):
            w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
            b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b, start + [nf])
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    """
    注意力层
    :param x: 原始输入 [batch,seq_len,embedding_size]
    :param scope: 变量域名
    :param n_state: 词向量维度
    :param past: 上文
    :param hparams: 超参数
    :return:
    """
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        """
        将输入文本的对应词向量分割，并转化为[batch, heads,sequence,features]的形式
        :param x:
        :return:
        """
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        x_reshape = split_states(x, hparams.n_head)
        return tf.transpose(x_reshape, [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        # 掩盖掉后面的序列
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        """
        计算多头注意力, q,k,v都是相同值经过不同矩阵线性变化而来
        :param q:
        :param k:
        :param v:
        :return:
        """
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        rsqrt = tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
        w = w * rsqrt

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state * 3)
        c_split = tf.split(c, 3, axis=2)
        q, k, v = map(split_heads, c_split)
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        # 词向量维度
        nx = x.shape[-1].value
        # 通过transorfrom计算
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        # 残差网络
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx * 4, hparams=hparams)
        x = x + m
        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]


def expand_tile(value, size):
    """
    根据传入的数据生成下标
    :param value:长度
    :param size: 每批数据量大小
    :return:
    """
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    """
    生成下标
    :param tokens: 当前文本
    :param past_length: 上一段文本长度
    :return:
    """
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    rt = expand_tile(past_length + tf.range(nsteps), batch_size)
    return rt


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        with tf.device('/cpu:0'):
            # 位置向量
            wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                                  initializer=tf.random_normal_initializer(stddev=0.01))
            # 词向量
            wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                                  initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]

        # 原始词向量与位置向量相加
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, 0))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')
        h_flat = tf.reshape(h, [batch, sequence, hparams.n_embd])
        results['h_flat'] = h_flat
        results['wte'] = wte
        return results


def parser(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([150], tf.int64),
                                           'title': tf.FixedLenFeature([20], tf.int64),
                                           'mask': tf.FixedLenFeature([1], tf.int64),
                                           'index': tf.FixedLenFeature([6], tf.int64),
                                           'tag': tf.FixedLenFeature([6], tf.int64)
                                       }
                                       )
    return features['seq'], features['title'], features['mask'], features['index'], features['tag']


def parser_dev(record):
    features = tf.parse_single_example(record,
                                       features={
                                           'seq': tf.FixedLenFeature([150], tf.int64),
                                           'title': tf.FixedLenFeature([20], tf.int64),
                                           'index': tf.FixedLenFeature([6], tf.int64)
                                       }
                                       )
    return features['seq'], features['title']


class GPT_2(object):

    def __init__(self, config):
        self.config = config
        self.__createModel()
        self.log_writer = tf.summary.FileWriter('../log/20190511214400', self.graph)
        self.train_data = {}
        self.test_data = {}

    # 合并所有tower上的梯度，取平均， 对于单机多卡程序，这段代码是通用的
    def __average_tower_grads(self, tower_grads):
        print('towerGrads:')
        idx = 0
        for grads in tower_grads:  # grads 为 一个list，其中元素为 梯度-变量 组成的二元tuple
            print('grads---tower_%d' % idx)
            idx += 1

        if len(tower_grads) == 1:
            return tower_grads[0]
        avgGrad_var_s = []
        for grad_var_s in zip(*tower_grads):
            grads = []
            v = None
            for g, v_ in grad_var_s:
                g = tf.expand_dims(g, 0)
                grads.append(g)
                v = v_
            all_g = tf.concat(grads, axis=0)
            avg_g = tf.reduce_mean(all_g, 0, keep_dims=False)
            avgGrad_var_s.append((avg_g, v));
        return avgGrad_var_s

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
        iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        seq, tag, mask = iter.get_next()

        seq = tf.cast(seq, tf.int32)
        tag = tf.cast(tag, tf.int32)
        mask = tf.cast(mask, tf.int32)

        seq = tf.reshape(seq, [-1, self.config.n_ctx])
        tag = tf.reshape(tag, [-1,20])
        mask = tf.reshape(mask, [-1])
        mask = tf.one_hot(mask, self.config.n_ctx)
        # 创建tag

        return seq, tag, mask, iter.make_initializer(dataset)

    def __train(self, seq, tag, mask, is_train=True):
        """
        计算损失函数
        :param seq: 序列
        :param tag: 预测序列
        :param mask: 掩码，去除无效部分
        :return:
        """
        result = model(self.config, seq, None, "encode", tf.AUTO_REUSE)
        h_flat1 = result['h_flat']
        result = model(self.config, tag, h_flat1, 'decode', tf.AUTO_REUSE)
        h_flat2 = result['h_flat']
        # 降维
        ft = tf.concat([tf.reduce_max(h_flat2, axis=1), tf.reduce_mean(h_flat2, axis=1)], axis=-1)

        # 选取候选词的表征
        h_flat1*mask

        ft = tf.layers.dropout(ft, 0.8, training=is_train)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(ft, self.config.num_classes, name='output-dense', reuse=tf.AUTO_REUSE)

        tag = tf.one_hot(tag, self.config.num_classes)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tag)

        p = tf.nn.softmax(logits)
        y_pred_cls = tf.argmax(p, 1)

        correct_pred = tf.equal(tf.argmax(tag, 1), y_pred_cls)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return tf.reduce_mean(loss), acc, p

    def __createModel(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            seq, tag, self.mask, self.train_data_op = self.__get_data(self.config.train_data_path, parser,
                                                                      is_train=True)
            dev_seq, dev_tag, self.dev_mask, self.dev_data_op = self.__get_data(self.config.test_data_path, parser)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate * 0.1)
            optimizer2 = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate,
            #                                       momentum=0.99, use_nesterov=True)
            loss, acc, _ = self.__train(seq, tag, self.mask)
            dev_loss, dev_acc, self.dev_p = self.__train(dev_seq, dev_tag, self.dev_mask, is_train=False)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optim = optimizer.minimize(loss, var_list=tf.trainable_variables()[2:-2])
                self.optim2 = optimizer2.minimize(loss, var_list=tf.trainable_variables()[-2:])

            self.loss = loss
            self.dev_acc = dev_acc
            self.dev_loss = dev_loss
            self.acc = acc
            self.dev_tag = dev_tag

            self.summary_train_loss = tf.summary.scalar('train_loss', self.loss)
            self.summary_train_acc = tf.summary.scalar('acc', self.acc)

            self.summary_dev_loss = tf.summary.scalar('dev_loss', self.dev_loss)
            self.summary_dev_acc = tf.summary.scalar('dev_acc', self.dev_acc)

            self.saver = tf.train.Saver()
            self.saver_v = tf.train.Saver(tf.trainable_variables()[:-2])

            self.merged = tf.summary.merge_all()

    def train(self, load_path, save_path):
        print('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次

        # 获取数据量
        test_data_count = get_data_count(self.config.test_data_path)
        train_data_count = get_data_count(self.config.train_data_path)
        require_improvement = (train_data_count // self.config.batch_size) * 10000  # 如果超过指定轮未提升，提前结束训练
        all_loss = 0
        all_acc = 0

        flag = False
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            if load_path != None:
                # sess.run(tf.global_variables_initializer())
                self.saver.restore(sess, load_path)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(self.config.num_epochs):
                if flag:
                    break
                print('Epoch:', epoch + 1)
                sess.run(self.train_data_op)
                for step in tqdm(range(train_data_count // self.config.batch_size + 1)):
                    if total_batch % self.config.print_per_batch == 0:
                        if total_batch % self.config.dev_per_batch == 0:
                            # 跑验证集
                            dev_loss, dev_acc = self.evaluate(sess, total_batch // self.config.dev_per_batch - 1,
                                                              test_data_count)
                            if min_loss == -1 or min_loss <= dev_acc:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_acc
                            else:
                                improved_str = ''
                                # self.saver.save(sess=sess, save_path=('./model/%d/model.ckpt'%(total_batch)))

                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train acc: {2:>6.5}, Val Loss: {3:>6.2}, ' \
                                  'Val acc: {4:>6.5}, Time: {5} {6}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch,
                                             all_acc / self.config.print_per_batch, dev_loss, dev_acc,
                                             time_dif, improved_str))
                        else:
                            time_dif = self.get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>6.2}, Time: {3}'
                            print(msg.format(total_batch, all_loss / self.config.print_per_batch,
                                             all_acc / self.config.print_per_batch, time_dif))
                        all_loss = 0
                        all_acc = 0

                    loss_train, train_acc, summary_loss, summary_acc, _, _ = sess.run(
                        [self.loss, self.acc, self.summary_train_loss,
                         self.summary_train_acc, self.optim2, self.optim])  # 运行优化
                    self.log_writer.add_summary(summary_loss, total_batch)
                    self.log_writer.add_summary(summary_acc, total_batch)
                    all_loss += loss_train
                    all_acc += train_acc
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        # 验证集正确率长期不提升，提前结束训练
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break  # 跳出循环
                if flag:
                    break

    def evaluate(self, sess, count, test_data_count):
        sess.run(self.dev_data_op)
        all_loss = 0
        all_acc = 0
        dev_count = 0
        # 1078209
        data_size = test_data_count // self.config.batch_size + 1
        for step in range(data_size):
            loss, acc, summary_loss, summary_acc = sess.run([self.dev_loss, self.dev_acc, self.summary_dev_loss,
                                                             self.summary_dev_acc])

            if not math.isnan(loss):
                self.log_writer.add_summary(summary_loss, count * data_size + step)
                self.log_writer.add_summary(summary_acc, count * data_size + step)
                all_loss += loss
                all_acc += acc
                dev_count += 1
        return all_loss / dev_count, all_acc / dev_count

    def p(self, load_path):
        rt = []
        tag = []
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self.saver.restore(sess, load_path)
            sess.run(self.dev_data_op)
            test_data_count = get_data_count(self.config.test_data_path)
            for step in tqdm(range(test_data_count // self.config.batch_size + 1)):
                p_tag, label = sess.run([self.dev_p, self.dev_tag])  # 运行优化
                rt.extend(p_tag.tolist())
                tag.extend(label.tolist())
        return {'p': rt, 'tag': tag}

    def get_time_dif(self, start_time):
        """获取已使用时间"""
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))


class Config(object):
    """CNN配置参数"""
    n_vocab = 382835  # 词库大小
    n_ctx = 150  # 序列最大长度
    n_embd = 384  # 词向量维度
    n_head = 6  # 注意力头数
    n_layer = 6  # 网络层数
    num_sampled = 8192
    num_classes = 562

    batch_size = 80 * 1
    learning_rate = 1e-3

    print_per_batch = 100  # 每多少轮输出一次结果
    dev_per_batch = 100  # 多少轮验证一次

    train_data_path = '../data/gpt_raw/20190621103415/100/fd/train_pd.record'
    test_data_path = '../data/gpt_raw/20190621103415/100/fd/dev_pd.record'
    dev_data_path = '../data/gpt_raw/20190621103415/100/fd/dev_pd.record'
    num_epochs = 200000


if __name__ == '__main__':
    config = Config()
    oj = GPT_2(config)
    with oj.graph.as_default():
        for index, x in enumerate(tf.trainable_variables()):
            print('%d:%s' % (index, x))
    path = '../model/20190615025600/model.ckpt'
    path_s = '../model/20190619105200/model.ckpt'
    oj.train(path, path_s)

    """
    20190613171549 0.55143
    20190614224200 0.63351
    20190615025600 0.64445

    20190612175435 0.45 使用数据：20190612175435  # 迁移后在准确标注数据训练的模型

    20190619105200 0.44939 使用数据：20190621103415

    20190618095200 0.56508 1000
    """