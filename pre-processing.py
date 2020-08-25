import os 
import pandas as pd 
import numpy as np
import argparse
import json
from sklearn.model_selection import train_test_split

def create_dataset(root_path):
    csv_file = os.path.join(root_path, 'data.csv')
    files = ['5g_corona_conspiracy', 'non_conspiracy', 'other_conspiracy']
    data_map = {
        'ID': [],
        'Text': [],
        'Label': []
    }
    for i in range(len(files)):
        path = os.path.join(root_path, files[i] + '.json')
        with open(path, 'r') as f:
            data = json.load(f)
        for j in range(len(data)):
            data_map['ID'].append(data[j]['id_str'])
            data_map['Text'].append(data[j]['full_text'])
            if i == 0:
                data_map['Label'].append(1)
            elif i == 1:
                data_map['Label'].append(3)
            elif i == 2:
                data_map['Label'].append(2)
    
    df = pd.DataFrame(data_map, columns=['ID', 'Text', 'Label'])
    df.to_csv(csv_file, index=False)
    print('Done')

def split(root_path):
    data_path = os.path.join(root_path, 'data.csv')
    train_file = os.path.join(root_path, 'train_data.csv')
    test_file = os.path.join(root_path, 'test_data.csv')
    df = pd.read_csv(data_path)
    rows = len(df.axes[0])
    X = df[df.columns[:2]]
    y = df['Label']
    df_train, df_test = train_test_split(df, test_size=0.2)
    df_train.to_csv(train_file, index=False)
    df_test.to_csv(test_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', help='Path to the Dataset')
    args = parser.parse_args()

    create_dataset(args.root)
    split(args.root)

if __name__ == "__main__":
    main()