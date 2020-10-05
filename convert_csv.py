import os 
import pandas as pd 
import numpy as np
import argparse
import json
import random
import csv

random.seed(2104)

TRAIN = 'train_train'
VAL = 'train_val'

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='Path to the Dataset')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='Ratio of the train set')
parser.add_argument('--out', help='Path to data folder', default='./data')
args = parser.parse_args()

# csv_file = os.path.join(args.root, 'data.csv')
root_path = args.root
ratio = args.ratio
out_path = args.out
labels = ['5g_corona_conspiracy', 'other_conspiracy', 'non_conspiracy']

data_map = {

}

for i in range(len(labels)):
    data_map.setdefault(labels[i], [])
    path = os.path.join(root_path, labels[i] + '.json')
    with open(path, 'r') as f:
        data = json.load(f)
    for j in range(len(data)):
        id_str = data[j]['id_str']
        text = data[j]['full_text']
        label = i + 1
        data_map[labels[i]].append((id_str, text, label))

data_out = [['ID', 'Text', 'Label']]
for k, v in data_map.items():
    data_out.extend([
        [id_str, text, label] for id_str, text, label in v
    ])

# Save data
if not os.path.isdir('./data'):
    os.system('mkdir ./data') 

csv.writer(open(f'{out_path}/data.csv', 'w')).writerows(data_out)
print('COMPLETE CONVERTING TO CSV FILE.')
