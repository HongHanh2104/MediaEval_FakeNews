import torch
import torch.nn as nn
import transformers
from tqdm import tqdm

__all__ = ['BaselineBERT', 'bert_base']

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
            for p in self.bert.parameters():
                p.requires_grad = False
        if freeze_embeddings:
            self.bert.embeddings.requires_grad = False
        self.drop = nn.Dropout(p=drop_p)
        self.out = nn.Linear(self.bert.config.hidden_size, nclasses)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.out(self.drop(pooled_output))

class bert_base(nn.Module):
    """Baseline model"""
    def __init__(self, nclasses):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        #print(self.bert)
        #self.bert.resize_token_embeddings(30525) 
        #for p in self.bert.parameters():
        #    p.requires_grad = False
        #self.bert.embeddings.requires_grad = False
        #self.bert = transformers.BertForPreTraining.from_pretrained('digitalepidemiologylab/covid-twitter-bert')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(1024, nclasses)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 512, bidirectional=True, batch_first=True)
    
    def forward(self, input_ids, attention_mask):
        #hs = self.bert.embeddings(input_ids, attention_mask)
        hs, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hs = self.drop(hs)
        hs, _ = self.lstm(hs)
        h = hs.mean(1)
        return self.out(h)

def main():
    model = SentimentClassifier(3)
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    model = model.to(device)
    print(model)

if __name__ == "__main__":
    main()
