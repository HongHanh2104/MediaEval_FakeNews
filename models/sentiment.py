import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

__all__ = ['BaselineBERT', 'TwitterBERT']

def set_freeze_module(m, state=False):
    for p in m.parameters():
        p.requires_grad = state

class BaselineBERT(nn.Module):
    def __init__(self, 
                 nclasses, 
                 version='bert-base-uncased', 
                 drop_p=0.3, 
                 freeze_bert=False, 
                 freeze_embeddings=False):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(version)
        if freeze_bert:
            set_freeze_module(self.bert)
        if freeze_embeddings:
            set_freeze_module(self.bert.embeddings)
        self.drop = nn.Dropout(p=drop_p)
        self.out = nn.Linear(self.bert.config.hidden_size, nclasses)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.out(self.drop(pooled_output))

class TwitterBERT(nn.Module):
    def __init__(self, 
                 nclasses,
                 freeze_bert=False, 
                 freeze_embeddings=False):
        super().__init__()
        self.bert = transformers.BertForPreTraining.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
        self.bert.cls.seq_relationship = nn.Linear(1024, nclasses, bias=True)
        if freeze_bert:
            set_freeze_module(self.bert)
        if freeze_embeddings:
            set_freeze_module(self.bert.embeddings)
    
    def forward(self, input_ids, attention_mask):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[1]

def main():
    model = SentimentClassifier(3)
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    model = model.to(device)
    print(model)

if __name__ == "__main__":
    main()
