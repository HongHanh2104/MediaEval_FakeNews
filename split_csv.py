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
data = df[['ID', 'Label', 'Text', 'hashtag', 'Cleaned_Text']].values

d = dict()
for id_str, lb, txt, hashtag, clean_txt in data:
    d.setdefault(lb, [])
    d[lb].append((id_str, txt, hashtag, clean_txt))

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
    out = [['ID', 'Label', 'Text', 'hashtag', 'Cleaned_Text']]
    out.extend([
        [id_str, lb, txt, hashtag, cleaned_txt]
        for lb, values in labels.items()
        for id_str, txt, hashtag, cleaned_txt in values
    ])
    csv.writer(open(f'{args.out}/{split}.csv', 'w')).writerows(out)

