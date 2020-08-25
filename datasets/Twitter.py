import torch 
from torch.utils import data
from torchvision import transforms as tvtf
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import argparse
import os 
from pathlib import Path

class Twitter(data.Dataset):
    def __init__(self,
                 tokenizer,
                 root_path=None,
                 train_file=None,
                 is_train=True,
                 max_len=200):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.root = Path(root_path)
        data_path = self.root / train_file
        self.data = pd.read_csv(data_path)
        #self.ids = data['ID']
        self.texts = self.data['Text']
        self.labels = self.data['Label']

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label).long()
        }


    def __len__(self):
        return len(self.data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--train', default='train_data.csv')
    args = parser.parse_args()

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    twitter = Twitter(tokenizer, args.root, args.train)
    print(twitter.__getitem__(47))



if __name__ == "__main__":
    main()