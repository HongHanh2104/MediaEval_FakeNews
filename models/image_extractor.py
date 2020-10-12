import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm

class ImageClassifier(nn.Module):
    arch = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2,
    }
    def __init__(self,
                 version,
                 nclasses):
        super().__init__()
        cnn = ImageClassifier.arch[version](pretrained=True)
        self.feature_dim = cnn.fc.in_features
        self.model = nn.Sequential(*list(cnn.children())[:-1])
        self.classifier = nn.Linear(self.feature_dim, nclasses)

    def forward(self, x):
        #print(x.shape)
        features = self.model(x)
        #print(features.shape)
        out = features.view(-1, self.feature_dim)
        #print(out.shape)
        out = self.classifier(out)
        #print(out.shape)
        return out

def main():
    model = ImageClassifier('resnet50', 3)
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    model = model.to(device)
    print(model)

if __name__ == "__main__":
    main()
