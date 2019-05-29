from data_process.util import *
import os
import json

"""将excel的数据按行写入txt"""

# data=read_excel('../data/处理后的数据合集(除去训练数据较少的类别)-现象.xlsx')[1:]

def get_index(indexs, keys):
    for index in indexs:
        cache = index.split('-')
        if cache[-1] in keys and cache[0] in keys:
            return cache


file_names=os.listdir('E:/data/trainData/通用语料')

indexs = ['content', 'title-answer', 'title-content']
with open('E:/data/trainData/通用语料/通用语料.txt', 'w', encoding='utf-8') as w:
    for name in file_names:
        with open('E:/data/trainData/通用语料/'+name,'r',encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                index = get_index(indexs,line.keys())
                if len(index) == 2:
                    s = line[index[0]].strip().replace('\r', '').replace('\n', '') \
                        + line[index[1]].strip().replace('\r', '').replace('\n', '') + '\n'
                else:
                    s = line[index[0]].strip().replace('\r', '').replace('\n', '') + '\n'
                w.write(s)

"""
with open('../data/处理后的数据合集.txt','w',encoding='utf-8') as f:
    for x in data:
        cache = [w for w in x[1]]
        f.write(x[1].strip().replace('\r','').replace('\n','')+'\n')
        f.write(x[-1] + '\n')
"""