import re
from util.tools import read_json
import json
import os
from tqdm import tqdm


if __name__=='__main__':
    data = read_json('../data/神箭关键词提取/docs.json')
    data.extend(read_json('../data/神箭关键词提取/train_data.json'))

    # 提取书名号内的词
    dicts = []
    for x in tqdm(data):
        content = x['content']+x['title']
        ret = re.findall("《(.+?)》", content)
        ret.extend(re.findall("“(.+?)”", content))
        ret.extend(re.findall("【(.+?)】", content))
        dicts.extend(ret)

    dicts = list(set(dicts))

    print('剔除过长文本')
    cache = dicts
    dicts=[]
    for x in cache:
        if len(x) < 10:
            dicts.append(x)

    print('提取词汇：%d'%(len(dicts)))

    print('开始合并词典')
    paths = os.listdir('../data/神箭关键词提取/字典')
    for path in paths:
        with open('../data/神箭关键词提取/字典/%s'%(path), 'r', encoding='utf-8') as f:
            for line in f:
                dicts.append(line.strip())
    dicts = list(set(dicts))
    print('词库大小：%d' % (len(dicts)))

    with open('../data/神箭关键词提取/user_dicts.txt','w', encoding='utf-8') as f:
        for x in dicts:
            f.write(x+' 200 n\n')
