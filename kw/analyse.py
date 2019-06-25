from jieba import analyse
import jieba
from util.tools import read_json,write_json
import re

def filter_char(sentence):
    return re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！：:，。？、~@#【】￥%……&*（）-]+", "",sentence)

jieba.load_userdict('../data/神箭关键词提取/user_dicts.txt')
tfidf = analyse.extract_tags

data = read_json('../data/神箭关键词提取/train_data.json')


rt = []
for x in data:
    cache ={'title':x['title'],'keywords':x['keywords'],'tfidf':tfidf(x['title']+'。'+x['content'],topK=20),'content':' '.join(list(jieba.cut(x['title']+'。'+x['content'])))}
    rt.append(cache)

count1 = 0
count2 = 0

for x in rt:
    add_one = 0
    for k in x['keywords'].split(','):
        if k in x['tfidf']:
            add_one+=1
    x['p'] = 'zero'
    if add_one!=0:
        x['p'] = 'of'
        count1+=1
    if add_one==len(x['keywords'].split(',')):
        x['p'] = 'all'
        count2 += 1

print('全部关键词在候选词的比例:%f'%(count2/len(rt)))

print('部分关键词在候选词的比例:%f'%(count1/len(rt)))

write_json(rt, '../data/神箭关键词提取/analyse/train_data_analyse.json')

# 分析文本长度

title = []
content = []
with open('../data/神箭关键词提取/all_docs.txt','r', encoding='utf-8') as f:
    for line in f:
        line = line.split('')
        title.append(len(list(jieba.cut(filter_char(line[1])))))
        content.append(len(list(jieba.cut(filter_char(line[2])))))

print(max(title))
print(max(content))

