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
from PIL import Image

import nlpaug.augmenter.word as naw

def random_aug(aug, text, p):
    if np.random.rand() < p:
        return aug(text)
    return text

class Twitter(data.Dataset):
    def __init__(self,
                 data_root_dir=None,
                 img_path=None,
                 img_size=224,
                 pretrain='bert-base-uncased',
                 max_len=128,
                 include_image=True, 
                 is_train=True):
        super().__init__()

        self.max_len = max_len
        self.data_dir = Path(data_root_dir)
        
        self.data = pd.read_csv(self.data_dir)
        #self.data = self.data[(self.data['Label'] != 2).values]
        
        self.ids = self.data['ID'].values
        self.texts = self.data['Text'].values
        self.labels = self.data['Label'].values
        self.tokenizer = self.get_tokenizer(pretrain, max_len)
        self.include_image = include_image
        self.is_train = is_train

        if self.include_image:
            self.image_path = Path(img_path)
            self.image_size = img_size

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
        #self.tokenizer.add_tokens(['5g', 'coronavirus', 'covid'])
        #self.init_augmenter()

    def init_augmenter(self):
        self.aug_insert = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert', device='cuda')
        self.aug_subs = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='substitute', device='cuda')

    def augment(self, text):
        return self.aug_insert.augment(text)
        text = random_aug(self.aug_insert.augment, text, 0.5)
        text = random_aug(self.aug_subs.augment, text, 0.5)
        return text

    def __getitem__(self, idx):
        if self.include_image:
            image = self.ids[idx]
            path = self.image_path / (str(image) + '.png')
            image = Image.open(path).convert('RGB')
            image = self.transforms(image)

        text = self.texts[idx]
        #if self.is_train:
        #    text = self.augment(text)
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
        if self.include_image:
            return ({
                'image': image,
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                }, torch.tensor(label).long())

        else:
            return ({
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                }, torch.tensor(label).long())

    def __len__(self):
        return len(self.ids)


class twitter_bert(Twitter):
    def __init__(self, data_root_dir, max_len=128, is_train=True):
        super(twitter_bert, self).__init__(
            data_root_dir, max_len=max_len, is_train=is_train
        )

    def get_tokenizer(self, pretrain='bert-base-uncased', max_len=128):
        return transformers.BertTokenizer.from_pretrained(pretrain, max_len=max_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    args = parser.parse_args()

    twitter = Twitter(args.root)
    #print(twitter.__getitem__(47))


if __name__ == "__main__":
    main()
