import pandas as pd
import csv
import argparse
import random
import os

print('Make sure your CSV has the first column for ID and the second column for class label')

TRAIN = 'train_train'
VAL = 'train_val'

parser = argparse.ArgumentParser()
parser.add_argument('--text', type=str,
                    help='Path to csv of text file')
parser.add_argument('--img', type=str,
                    help='Path to folder of images')
parser.add_argument('--ratio', type=float, default=0.8,
                    help='ratio of the train set (default: 0.8)')
parser.add_argument('--seed', type=int, default=2104,
                    help='random seed (default: 2104)')
parser.add_argument('--out', type=str, default='.',
                    help='directory to save the splits (default: .)')

args = parser.parse_args()

# Seed the random processes
random.seed(args.seed)

# Load CSV
df = pd.read_csv(args.text)
data = df.values

d = dict()
for id_str, lb, *metadata in data:
    d.setdefault(lb, [])
    path = os.path.join(args.img, str(id_str) + '.png')
    if os.path.isfile(path):
        d[lb].append(id_str)


out = [['ID', 'Label']]
out.extend([
    [id_str, lb]
    for lb, value_list in d.items()
    for id_str in value_list
])

csv.writer(open(f'{args.out}/{os.path.basename(args.text)}', 'w')).writerows(out)

