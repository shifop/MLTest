import numpy as np


def p(h_v, g_v):
    """
    使用动态规划找到最大可能性的标注
    :param h_v: 词-词性得分矩阵 [batch, seq_length, pos_size]
    :param g_v: 词性-词性转移得分矩阵 [pos_size, pos_size]
    :return:
    """
    pos_size = g_v.shape[0]
    seq_length = h_v.shape[1]
    batch_size = h_v.shape[0]
    path = [[[] for x in range(pos_size)] for i in range(batch_size)]  # 在当前阶段，标注为不同词性的最大得分的序列
    score = np.zeros([batch_size, seq_length, pos_size], np.float32)  # 在当前阶段，标注为对应词性的最大得分

    score[:, 0, :] += h_v[:, 0, :]
    for index in range(batch_size):
        for i in range(pos_size):
            path[index][i].append(i)
    for index in range(1, seq_length):
        # 计算在当前阶段，标注为不同词性的最大得分
        path_cache = path
        path = [[[] for x in range(pos_size)] for i in range(batch_size)]
        for i in range(pos_size):
            cache = np.array([score[:, index - 1, y] + g_v[y, i] + h_v[:, index, i] for y in range(pos_size)])
            max_v = cache.max(axis=0)
            max_index = cache.argmax(axis=0)
            # 更新得分
            score[:, index, i] = max_v
            # 更新路径
            for x in range(batch_size):
                path[x][i].extend(path_cache[x][i])
                path[x][i].append(max_index[x])

    max_index = score[:,seq_length-1,:].argmax(axis=-1)
    rt=[]
    for index,x in enumerate(max_index):
        rt.append(path[index][x])
    return rt


if __name__=='__main__':
    h_v = np.random.rand(32,10,44)
    g_v = np.random.rand(44,44)
    p(h_v,g_v)
