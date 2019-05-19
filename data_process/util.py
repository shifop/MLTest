import xlrd
import json
import re
import xlsxwriter

'''
处理数据，训练数据
'''


def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def filter_char(sentence):
    return re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！：:，。？、~@#【】￥%……&*（）-]+", "",sentence)


def read_excel(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]          #通过索引顺序获取
    rows=table.nrows
    rt=[]
    for x in range(rows):
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