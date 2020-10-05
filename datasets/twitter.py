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
        #self.data = self.data[(self.data['Label'] != 2).values]
        
        self.ids = self.data['ID'].values
        self.texts = self.data['Cleaned_Text'].values
        self.labels = self.data['Label'].values
        self.tokenizer = self.get_tokenizer('bert-base-uncased')

        #self.tokenizer.add_tokens(['5g', 'coronavirus', 'covid'])

    def get_tokenizer(self, pretrain=None):
        if pretrain == None:
            return transformers.BertTokenizer
        return transformers.BertTokenizer.from_pretrained(pretrain)

    def __getitem__(self, idx):
        text = self.texts[idx]
        #label = 0 if self.labels[idx] < 2 else 1
        label = self.labels[idx] - 1
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        #print(self.tokenizer.tokenize(text), label)

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label).long()
        }


    def __len__(self):
        return len(self.ids)

class twitter_bert(Twitter):
    def __init__(self, data_root_dir, max_len=128, is_train=True):
        super(twitter_bert, self).__init__(
            data_root_dir, max_len=max_len, is_train=is_train
        )

    def get_tokenizer(self, pretrain=None):
        return transformers.BertTokenizer.from_pretrained('bert-base-uncased', max_len=128)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    args = parser.parse_args()

    twitter = Twitter(args.root)
    #print(twitter.__getitem__(47))



if __name__ == "__main__":
    main()
