import pandas as pd
import csv
import argparse
import random

print('Make sure your CSV has the first column for ID and the second column for class label')

TRAIN = 'train_train'
VAL = 'train_val'

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str,
                    help='path to csv file')
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
df = pd.read_csv(args.csv)
data = df.values

d = dict()
for id_str, lb, *metadata in data:
    d.setdefault(lb, [])
    d[lb].append((id_str, *metadata))

splits = {
    TRAIN: dict(),
    VAL: dict(),
}

for lb, value_list in d.items():
    train_sz = max(int(len(value_list) * 0.8), 1)
    shuffled = random.sample(value_list, k=len(value_list))
    splits[TRAIN][lb] = shuffled[:train_sz]
    splits[VAL][lb] = shuffled[train_sz:]

# Split
for split, labels in splits.items():
    out = [list(df.keys())]
    out.extend([
        [id_str, lb, *metadata]
        for lb, values in labels.items()
        for id_str, *metadata in values
    ])
    csv.writer(open(f'{args.out}/{split}.csv', 'w')).writerows(out)

