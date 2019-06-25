import xlrd
import json
import re
import xlsxwriter
from tqdm import tqdm

'''
处理数据，训练数据
'''


def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def write_json(data,path):
    with open(path,'w',encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))


def filter_char(sentence):
    return re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！：:，。？、~@#【】￥%……&*（）-]+", "",sentence)


def read_excel(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]          #通过索引顺序获取
    rows=table.nrows
    rt=[]
    for x in tqdm(range(rows)):
        cache=table.row_values(x)
        for index in range(len(cache)):
            if not isinstance(cache[index],str):
                cache[index]=str(int(cache[index]))
        rt.append(cache)
    return rt


def write_excel(path,data):
    workbook = xlsxwriter.Workbook(path)     #创建工作簿
    worksheet = workbook.add_worksheet()            #创建工作表
    for index,x in enumerate(data):
        for j,cell_v in enumerate(x):
            worksheet.write(index,j,cell_v)
    workbook.close()


def read_text(path):
    rt = []
    with open(path, 'r', encoding='utf-8') as f:
        for content in tqdm(f):
            content = content.strip()
            if not content:
                continue
            rt.append(content)
    return rt
