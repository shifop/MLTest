import json
import random
from tqdm import tqdm
import jieba.posseg as pseg
from data_process.util import read_json

"""创建词典，过滤频率较低的词汇"""


def is_alphabet(uchar):
    if (u'\u0041' <= uchar<=u'\u005a') or (u'\u0061' <= uchar<=u'\u007a'):
        return True
    else:
        return False

# dict = {}
# with open('../data/lm/0-5000000-cut.txt', 'r', encoding='utf-8') as f:
#     for content in tqdm(f):
#         content = content.strip()
#         if not content:
#             continue
#         content = content.split(' ')
#         for word in content:
#             if word == ' ' or word == '':
#                 continue
#             if word.isdigit():
#                 word = ('<%dNUM>' % (len(word)))
#             if word not in dict.keys():
#                 dict[word]=0
#             dict[word] += 1
#
#
# with open('../data/lm/通用语料-cut.txt', 'r', encoding='utf-8') as f:
#     for content in tqdm(f):
#         content = content.strip()
#         if not content:
#             continue
#         content = content.split(' ')
#         for word in content:
#             if word == ' ' or word == '':
#                 continue
#             if word.isdigit():
#                 word = ('<%dNUM>' % (len(word)))
#             if word not in dict.keys():
#                 dict[word]=0
#             dict[word] += 1
#
# dx_data = []
# with open('../data/lm/电信工单-语言模型训练语料-cut.txt', 'r', encoding='utf-8') as f:
#     for content in tqdm(f):
#         content = content.strip()
#         if not content:
#             continue
#         dx_data.append(content)
#
# random.shuffle(dx_data)
#
# dx_data = dx_data[:1000000]
#
# for content in dx_data:
#     content = content.split(' ')
#     for word in content:
#         if word == ' ' or word == '':
#             continue
#         if word.isdigit():
#             word = ('<%dNUM>' % (len(word)))
#         if word not in dict.keys():
#             dict[word] = 0
#         dict[word] += 1


dict = read_json('../data/lm/dict.json')

# 剔除出频率较低的词汇
cache = dict
dict = {}
delete = []
for word in cache.keys():
    if cache[word] >= 100:
        dict[word] = cache[word]
    else:
        delete.append(word)


# 数字，字母替换为<NUM>,<CHAR>,人名替换为<NAME>
# 1. 长词再次分词
# 2. 排列组合分解

print('dict count:%d, delete count:%d' % (len(dict),len(delete)))
dict['<NAME>'] = 0
bp = {}
for word in tqdm(delete):
    # 处理字母和数字
    if word.isdigit():
        if ('<%dNUM>'%(len(word))) not in dict.keys():
            dict[('<%dNUM>'%(len(word)))]=0
        dict[('<%dNUM>'%(len(word)))] += 1
        bp[word] = [('<%dNUM>'%(len(word)))]
        continue
    if is_alphabet(word):
        if ('<%dCHAR>'%(len(word))) not in dict.keys():
            dict[('<%dCHAR>'%(len(word)))]=0
        dict[('<%dCHAR>'%(len(word)))] += 1
        bp[word] = [('<%dCHAR>' % (len(word)))]
        continue

    # 再次分词
    bp[word] = []
    wordc = word
    word = [[x,y] for x,y in pseg.cut(word)]
    cache = []
    for x, y in word:
        if y == 'x':
            continue
        if y == 'nr':
            cache.append(['<NAME>', 1])
        elif x in dict.keys():
            cache.append([x, 1])
        else:
            cache.append([x, 0])

    # 寻找最大子串
    for c in cache:
        x = c[0]
        if c[1] == 1:
            bp[wordc].append(x)
            dict[x] += 1
            continue
        start_i = 0
        length = len(x)
        while length !=0:
            for index in range(len(x),start_i,-1):
                if x[start_i:index] in dict.keys():
                    dict[x[start_i:index]] += 1
                    bp[wordc].append(x[start_i:index])
                    start_i = index
                    length = len(x) - index
                    break
                if index == start_i+1:
                    dict[x[start_i:index]] = 1
                    bp[wordc].append(x[start_i:index])
                    start_i = index
                    length = len(x) - index


# 再次过滤频次低于5的
cache = dict
dict = {}
delete = []
for word in cache.keys():
    if cache[word] >= 100:
        dict[word] = cache[word]
    else:
        delete.append(word)

dict['<NUM>'] = 1
dict['CHAR'] = 1
dict['<START>'] = 1
dict['<END>'] = 1

w2i = {}
for index, word in enumerate(dict.keys()):
    w2i[word]= index

print('dict count:%d, delete count:%d' % (len(dict),len(delete)))

"""
with open('../data/lm/dict.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(dict, ensure_ascii=False))

"""
with open('../data/lm/w2i.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(w2i, ensure_ascii=False))

with open('../data/lm/delete.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(delete, ensure_ascii=False))

with open('../data/lm/bpe.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(bp, ensure_ascii=False))

# with open('../data/lm/电信工单-语言模型训练语料100w-cut.txt', 'w', encoding='utf-8') as f:
#     for x in dx_data:
#         s = x.strip().replace('\r', '').replace('\n', '') + '\n'
#         f.write(s)

