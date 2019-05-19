import json

with open('../data/data.json','r',encoding='utf-8') as f:
    data=json.loads(f.read())

with open('../data/data-s.json','w', encoding='utf-8') as f:
    f.write(json.dumps(data, ensure_ascii=False))

