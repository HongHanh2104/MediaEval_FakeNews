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
    'ID': [],
    'Text': [],
    'Label': []
}

for i in range(len(labels)):
    path = os.path.join(root_path, labels[i] + '.json')
    with open(path, 'r') as f:
        data = json.load(f)
    for j in range(len(data)):
        data_map['ID'].append(data[j]['id_str'])
        data_map['Text'].append(data[j]['full_text'])
        data_map['Label'].append(i + 1)



splits = {
    TRAIN: dict(),
    VAL: dict(),
}
# Split
d = data_map
for cls_id, cls_list in d.items():
    train_sz = max(int(len(cls_list) * 0.8), 1)
    shuffled = random.sample(cls_list, k=len(cls_list))
    splits[TRAIN][cls_id] = shuffled[:train_sz]
    splits[VAL][cls_id] = shuffled[train_sz:]

# Save data
os.system('mkdir ./data')
df = pd.DataFrame(data_map, columns=['ID', 'Text', 'Label'])
df.to_csv(f'{out_path}/data.csv', index=False)
for split, data in splits.items():
    df = pd.DataFrame(data, columns=['ID', 'Text', 'Label'])
    df.to_csv(f'{out_path}/{split}.csv', index=False)