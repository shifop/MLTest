from data_process.util import *

"""将excel的数据按行写入txt"""

data=read_excel('../data/处理后的数据合集(除去训练数据较少的类别)-现象.xlsx')[1:]

with open('../data/处理后的数据合集.txt','w',encoding='utf-8') as f:
    for x in data:
        cache = [w for w in x[1]]
        f.write(x[1].strip().replace('\r','').replace('\n','')+'\n')
        f.write(x[-1] + '\n')