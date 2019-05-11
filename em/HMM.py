import numpy as np
from tqdm import tqdm
import multiprocessing
import sys

"""基于HMM模型的词性标注
使用EM求解参数
"""


def get_map(data):
    rt = {}
    for index, x in enumerate(data):
        rt[x] = index
    return rt


class HMM(object):

    def __init__(self, words, pos, train_data):
        """
        初始化参数
        :param words: 词典
        :param pos: 词性列表
        :param train_data: 训练材料,[[词, 词],[词, 词],...]这样的格式
        """
        self.data = train_data
        self.w2i = get_map(words)  # 词-索引列表
        self.p2i = get_map(pos)  # 词性-索引列表

        """初始化参数"""
        # 初始词性列表
        self.m = np.random.normal(10, 5, len(pos))
        self.m = self.m / self.m.sum()

        # 词性-词性转移矩阵
        self.p2p = np.random.rand(len(pos), len(pos))
        self.p2p = self.p2p / self.p2p.sum(axis=1, keepdims=True)

        # 词性-词转移矩阵
        self.p2w = np.random.rand(len(pos), len(words))
        self.p2w = self.p2w / self.p2w.sum(axis=1, keepdims=True)

    def __fab(self, m, p2p, p2w, o):
        """
        计算在对应观测序列下，隐状态在不同位置上的期望
        :param m: 词性分布
        :param p2p: 词性-词性转移矩阵
        :param p2w: 词性-词转移矩阵
        :param o: 观测序列
        :return: 对应观测序列下，不同位置上隐状态的期望
        """
        pos_size = len(m)
        o_size = len(o)

        """计算不同词性在不同序列位置上的期望
        使用动态规划算法计算
        递推式：

        """
        f_m = np.zeros((pos_size, o_size), np.float64)  # f_m[i][j] 代表序列位置j上，词性是pos[i], 位置j以及j以前的词是o[:j]的期望
        # 计算初始状态
        for x in range(pos_size):
            f_m[x][0] = m[x] * p2w[x][self.w2i[o[0]]]
        # 计算在个序列位置上的值
        for index in range(1, o_size):
            for p_i in range(pos_size):
                cache = 0
                for p_ii in range(pos_size):
                    cache += f_m[p_ii][index - 1] * p2p[p_ii][p_i] * p2w[p_i][self.w2i[o[index]]]
                if cache==0:
                    cache = sys.float_info.min
                f_m[p_i][index] = cache

        b_m = np.zeros((pos_size, o_size), np.float64)  # f_m[i][j] 代表序列位置j上，词性是pos[i], 位置j以及j以前的词是o[:j]的期望
        # 计算初始状态
        for x in range(pos_size):
            b_m[x][-1] = m[x] * p2w[x][self.w2i[o[-1]]]
        # 计算在个序列位置上的值
        for index in range(o_size - 2, -1, -1):
            for p_i in range(pos_size):
                cache = 0
                for p_ii in range(pos_size):
                    cache += b_m[p_ii][index + 1] * p2p[p_ii][p_i] * p2w[p_i][self.w2i[o[index]]]
                if cache==0:
                    cache = sys.float_info.min
                b_m[p_i][index] = cache

        return np.array(f_m), np.array(b_m)

    def __up(self, m, p2p, p2w, Os):
        """
        直接根据前一步的参数计算下一步的更新值
        :param m: 各词性分布
        :param p2p: 词性-词性转移矩阵
        :param p2w: 词性-词转移矩阵
        :param Os: 训练数据
        :return:
        """
        pos_size = len(m)

        n_m = np.zeros(m.shape, np.float64)
        n_p2p_1 = np.zeros(p2p.shape, np.float64)
        n_p2p_2 = np.zeros(p2p.shape, np.float64)

        n_p2w_1 = np.zeros(p2w.shape, np.float64)
        n_p2w_2 = np.zeros(p2w.shape, np.float64)
        for o in tqdm(Os):
            f_m, b_m = self.__fab(m, p2p, p2w, o)

            # 在o观测序列下，不同词性在不同位置上的概率
            r = f_m * b_m + sys.float_info.min
            r = r / (r.sum(axis=0, keepdims=True))

            # 在观测序列下，不同词性在不同位置上的转移概率
            o_size = len(o)
            ks = np.zeros([o_size - 1, pos_size, pos_size])
            for index in range(o_size - 1):
                for p_i in range(pos_size):
                    for p_ii in range(pos_size):
                        cache = f_m[p_i][index] * p2p[p_i][p_ii] \
                                * p2w[p_ii][self.w2i[o[index]]] * b_m[p_ii][index + 1]
                        if cache==0:
                            cache = sys.float_info.min
                        ks[index][p_i][p_ii] = cache

            ks = ks + sys.float_info.min
            ks = ks / ks.sum(axis=2, keepdims=True).sum(axis=1, keepdims=True)

            """ 更新词性分布 """
            cache = np.array([r[index][0] for index in range(pos_size)])
            n_m = n_m + cache

            """ 更新词性-词性转移矩阵 """
            n_p2p_1 = n_p2p_1 + ks.sum(axis=0)
            n_p2p_2 = n_p2p_2 + r[:,:-1].sum(axis=1, keepdims=True)

            """ 更新词性-词转移矩阵 """
            for p_i in range(pos_size):
                for index in range(o_size):
                    cache = n_p2w_1[p_i][self.w2i[o[index]]] + r[p_i][index]
                    if cache==0:
                        print('_')
                    n_p2w_1[p_i][self.w2i[o[index]]] = cache
            n_p2w_2 = r.sum(axis=1, keepdims=True)

        n_m = n_m / len(Os)
        n_p2p = n_p2p_1 / (n_p2p_2 + sys.float_info.min)
        n_p2w = n_p2w_1 / (n_p2w_2 + sys.float_info.min)

        return n_m, n_p2p, n_p2w

    def train(self):
        """
        训练
        :return:
        """
        print('当前参数')
        print(self.m)
        print('更新参数')
        # pool = multiprocessing.Pool(processes=6)
        self.m, self.p2p, self.p2w = self.__up(self.m, self.p2p, self.p2w, self.data)
        print('更新后的参数')
        print(self.m)

    def __Viterbi(self, fs, s2s, s2v, o, tag, dicts):
        """
        使用参数：词性初始分布，词性转移矩阵，词性-词转移矩阵
        以及文本计算最大可能的词性序列
        """
        tags = [[[] for y in tag.keys()] for x in o]
        f = [[0 for y in tag.keys()] for x in o]  # 当前词性为，，，，的最大概率
        for index, x in enumerate(fs):
            f[0][index] = x * s2v[index][dicts[o[0]]]
        for x in range(len(tag)):
            tags[0][x] = [x]
        for i in range(1, len(o)):
            for x in range(len(f[i])):
                cache = [f[i - 1][p] * s2s[p][x] * s2v[x][dicts[o[i]]] for p in range(len(f[i - 1]))]
                f[i][x] = max(cache)
                index = cache.index(f[i][x])
                tags[i][x] = [t for t in tags[i - 1][index]]
                tags[i][x].append(x)

        #  返回最大概率的词性序列
        index = f[-1].index(max(f[-1]))
        tag_r = {}
        for x in tag.keys():
            tag_r[tag[x]] = x
        rt = [tag_r[x] for x in tags[-1][index]]
        return rt, f[-1][index]

    def p(self, o):
        """
        预测
        :return:
        """
        s, v = self.__Viterbi(self.m, self.p2p, self.p2w, o, self.p2i, self.w2i)

        return s


if __name__=="__main__":
    data = []
    with open('../data/词性标注@人民日报199801.txt', 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())

    cut_data = []
    for x in data[:100]:
        cut_data.append(x.split(' '))

    tag = []
    train_data = []
    for x in cut_data:
        words = []
        for y in x:
            cache = y.split('/')
            if len(cache) < 2:
                continue
            else:
                tag.append('/' + y.split('/')[1].split(']')[0])
                words.append(cache[0])
        if len(words) != 0:
            train_data.append(words)

    tag = list(set(tag))

    words = []
    for x in train_data:
        words.extend(x)

    words = list(set(words))

    oj = HMM(words, tag, train_data)

    for x in range(10):
        if x>0:
            print(x)
        oj.train()

