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
                 data_root_dir=None,
                 max_len=200, 
                 is_train=True):
        super().__init__()

        self.max_len = max_len
        self.data_dir = Path(data_root_dir)
        
        self.data = pd.read_csv(self.data_dir)
        
        self.ids = self.data['ID']
        self.texts = self.data['Text']
        self.labels = self.data['Label']
        self.tokenizer = self.get_tokenizer('bert-base-uncased')

    def get_tokenizer(self, pretrain=None):
        if pretrain == None:
            return transformers.BertTokenizer
        return transformers.BertTokenizer.from_pretrained(pretrain)

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
        return len(self.ids)

class twitter_bert(Twitter):
    def __init__(self, data_root_dir, max_len=200, is_train=True):
        super(twitter_bert, self).__init__(
            data_root_dir, max_len=max_len, is_train=is_train
        )

    def get_tokenizer(self, pretrain=None):
        return transformers.BertTokenizer.from_pretrained('bert-base-uncased', max_len=200)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    args = parser.parse_args()

    twitter = Twitter(args.root)
    #print(twitter.__getitem__(47))



if __name__ == "__main__":
    main()