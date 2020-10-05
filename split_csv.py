import pandas as pd
import csv
import argparse
import random

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
data = df[['ID', 'Cleaned_Text', 'Label']].values

# Build class to image_fns dictionary
d = dict()
for fn, txt, lb in data:
    d.setdefault(lb, [])
    d[lb].append((fn, txt))

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
    out = [['ID', 'Cleaned_Text', 'Label']]
    out.extend([
        [id_str, text, lb]
        for lb, values in labels.items()
        for id_str, text in values
    ])
    csv.writer(open(f'{args.out}/{split}.csv', 'w')).writerows(out)

