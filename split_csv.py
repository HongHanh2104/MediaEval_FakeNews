import pandas as pd
import csv
import argparse
import random

TRAIN = 'train_train'
VAL = 'train_val'

parser = argparse.ArgumentParser()
parser.add_argument('-csv', type=str,
                    help='path to csv file')
parser.add_argument('-ratio', type=float, default=0.8,
                    help='ratio of the train set (default: 0.8)')
parser.add_argument('-seed', type=int, default=3698,
                    help='random seed (default: 3698)')
parser.add_argument('-out', type=str, default='.',
                    help='directory to save the splits (default: .)')

args = parser.parse_args()

# Seed the random processes
random.seed(args.seed)

# Load CSV
df = pd.read_csv(args.csv)
data = df[['ID', 'Cleaned_Text', 'Label']].values

# Build class to image_fns dictionary
d = dict()
for fn, txt, cl in data:
    d.setdefault(cl, [])
    d[cl].append((fn, txt))

# Stratified split
splits = {
    TRAIN: dict(),
    VAL: dict(),
}
for cls_id, cls_list in d.items():
    train_sz = max(int(len(cls_list) * args.ratio), 1)
    shuffled = random.sample(cls_list, k=len(cls_list))
    splits[TRAIN][cls_id] = shuffled[:train_sz]
    splits[VAL][cls_id] = shuffled[train_sz:]

# Save split
for split, classes in splits.items():
    out = [['ID', 'Cleaned_Text', 'Label']]
    out.extend([
        [fn[0], fn[1], cl]
        for cl, fns in classes.items()
        for fn in fns
    ])
    csv.writer(open(f'{args.out}/{split}.csv', 'w')).writerows(out)
