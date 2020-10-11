import torch
from torch.utils.data import Dataset
import torchvision.transforms as tvtf

import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import os

import argparse

class TwitterImageDataset(Dataset):
    def __init__(self,
                 csv_data,
                 img_path,
                 img_size=224,
                 is_train=True):
        super().__init__()

        self.img_size = img_size
        self.image_path = Path(img_path)  
        self.data = pd.read_csv(Path(csv_data))
        self.images = self.data['ID'].values
        self.labels = self.data['Label'].values

        self.is_train = is_train

        if self.is_train:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = tvtf.Compose([
                tvtf.Resize((224, 224)),
                tvtf.ToTensor(),
                tvtf.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        path = self.image_path / (str(image) + '.png')
        image = Image.open(path)
        return self.transforms(image), label     

    def __len__(self):
        return len(self.images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv')
    parser.add_argument('--image')
    args = parser.parse_args()

    dataset = TwitterImageDataset(args.csv, args.image, classes=3)
    dataset.__getitem__(1)


if __name__ == "__main__":
    main()

    
        
