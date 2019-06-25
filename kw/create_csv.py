import csv
from data_process.util import read_json
from tqdm import tqdm

data =read_json('../data/神箭关键词提取/rt2.json')

with open('../data/神箭关键词提取/sub.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label1', 'label2'])
    for x in tqdm(data):
        k = x['keywords']
        if len(k) != 2:
            k.append('keywords')
        csv_writer.writerow([x['id'], k[0], k[1]])