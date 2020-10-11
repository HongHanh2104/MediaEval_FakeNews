import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm

class ImageClassifier(nn.Module):
    def __init__(self,
                 nclasses):
        super().__init__()
        cnn = models.resnet50(pretrained=True)
        self.feature_dim = cnn.fc.in_features
        self.model = nn.Sequential(*list(cnn.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, nclasses)

    def forward(self, x):
        #print(x.shape)
        features = self.model(x) # [B, D, H', W']
        #print(features.shape)
        out = features.view(-1, self.feature_dim)
        #print(out.shape)
        out = self.classifier(out)
        #print(out.shape)
        return out

def main():
    model = ImageClassifier(3)
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    model = model.to(device)
    print(model)

if __name__ == "__main__":
    main()
