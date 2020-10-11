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
LABELS = ['5g_corona_conspiracy', 'other_conspiracy', 'non_conspiracy']

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='Path to the Dataset')
parser.add_argument('--img', help='Path to Image folder')
parser.add_argument('--out_text', help='Path to text data folder', default='./data/text')
parser.add_argument('--out_img', help='Path to img data folder', default='./data/image')
args = parser.parse_args()

root_path = args.root
img_path = args.img
out_text = args.out_text
out_img = args.out_img

# Save data
if not os.path.isdir(out_text):
    os.system(f'mkdir {out_text}')
if not os.path.isdir(out_img):
    os.system(f'mkdir {out_img}') 

def create_text_data(root, out_text):
    text_map = {

    }

    for i in range(len(LABELS)):
        text_map.setdefault(LABELS[i], [])
        path = os.path.join(root_path, LABELS[i] + '.json')
        with open(path, 'r') as f:
            data = json.load(f)
        for j in range(len(data)):
            id_str = data[j]['id_str']
            text = data[j]['full_text']
            label = i + 1
            text_map[LABELS[i]].append((id_str, label, text))

    data_out = [['ID', 'Label', 'Text']]
    for k, v in text_map.items():
        data_out.extend([
            [id_str, label, text] for id_str, label, text in v
        ])

    csv.writer(open(f'{out_text}/data.csv', 'w')).writerows(data_out)
    print('COMPLETE CONVERTING TO TEXT CSV FILE.')

def create_img_data(root, img_path, out_img):
    img_map = {

    }

    for i in range(len(LABELS)):
        img_map.setdefault(LABELS[i], [])
        path = os.path.join(root_path, LABELS[i] + '.json')
        with open(path, 'r') as f:
            data = json.load(f)
        for j in range(len(data)):
            id_str = data[j]['id_str']
            img = os.path.join(img_path, id_str + ".png")
            if os.path.isfile(img):
                label = i + 1
                img_map[LABELS[i]].append((id_str, label))
    
    data_out = [['ID', 'Label']]
    for k, v in img_map.items():
        data_out.extend([
            [id_str, label] for id_str, label in v
        ])

    csv.writer(open(f'{out_img}/data.csv', 'w')).writerows(data_out)
    print('COMPLETE CONVERTING TO IMAGE CSV FILE.')

create_text_data(root_path, out_text)
create_img_data(root_path, img_path, out_img)