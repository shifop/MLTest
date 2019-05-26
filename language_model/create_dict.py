import json
"""创建词典，过滤频率较低的词汇"""

dict={}
with open('../data/lm/0-5000000-cut.txt', 'r', encoding='utf-8') as f:
    while 1:
        content = f.readline().strip()
        if not content:
            break
        content = content.split(' ')
        for word in content:
            if word==' ':
                continue
            if word not in dict.keys():
                dict[word]=0
            dict[word] += 1


with open('../data/lm/0-5000000-cut.txt', 'r', encoding='utf-8') as f:
    while 1:
        content = f.readline().strip()
        if not content:
            break
        content = content.split(' ')
        for word in content:
            if word==' ':
                continue
            if word not in dict.keys():
                dict[word]=0
            dict[word] += 1

with open('../data/lm/电信工单-语言模型训练语料-cut.txt', 'r', encoding='utf-8') as f:
    while 1:
        content = f.readline().strip()
        if not content:
            break
        content = content.split(' ')
        for word in content:
            if word==' ':
                continue
            if word not in dict.keys():
                dict[word]=0
            dict[word] += 1

# 将频率较低的词汇拆成子词
cache = dict
dict = {}
delete = []
for word in cache.keys():
    if cache[word]>=5:
        dict[word]=cache[word]
    else:
        delete.append(word)

for word in delete:

    for chart in word:
        if chart not in dict:
            dict[chart]=0
        dict[chart] += 1



with open('../data/lm/dict.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(dict))

