import os 
import pandas as pd 
import numpy as np
import argparse
import json
import random
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--json', help='Path to the JSON path')
parser.add_argument('--csv', help='Path to the CSV path', default='./data')
parser.add_argument('--file', help='A CSV File')
args = parser.parse_args()

json_path = args.json
csv_path = args.csv
csv_file = args.file

json_files = ['5g_corona_conspiracy', 'other_conspiracy', 'non_conspiracy']

data = pd.read_csv(os.path.join(csv_path, csv_file))

id_lb1 = list(data[data['Label'] == 1]['ID'])
id_lb2 = list(data[data['Label'] == 2]['ID'])
id_lb3 = list(data[data['Label'] == 3]['ID'])

csv_files = [id_lb1, id_lb2, id_lb3]

wrong = {

}
for i in range(len(json_files)):
    wrong.setdefault(i, [])
    path = os.path.join(json_path, json_files[i] + '.json')
    with open(path, 'r') as f:
        jdata = json.load(f)
    for j in csv_files[i]:
        if str(j) not in jdata:
            wrong[i].append(j)
print(wrong)
