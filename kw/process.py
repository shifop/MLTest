from util.tools import *
from tqdm import tqdm


data = read_text('../data/神箭关键词提取/all_docs.txt')
keywords = read_text('../data/神箭关键词提取/train_docs_keywords.txt')

cache = keywords
keywords=[]
for x in tqdm(cache):
    x = x.split('	')
    if len(x)!=2:
        print('error:%s'%(x))
    keywords.append({'id':x[0],'keywords':x[1]})

key_id_index=[x['id'] for x in keywords]

cache = data
data = []
train_data = {}
for x in tqdm(cache):
    x = x.split('')
    if len(x)!=3:
        print('error:%s'%(x))
    if x[0] not in key_id_index:
        data.append({'id':x[0],'title':x[1],'content':x[2]})
    else:
        train_data[x[0]]={'id':x[0],'title':x[1],'content':x[2]}

save_train_data = []
for x in tqdm(keywords):
    cache = train_data[x['id']]
    cache['keywords']=x['keywords']
    save_train_data.append(cache)

write_json(data,'../data/神箭关键词提取/docs.json')
write_json(save_train_data,'../data/神箭关键词提取/train_data.json')