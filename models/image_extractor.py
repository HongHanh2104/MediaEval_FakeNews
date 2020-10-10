import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm

class ImageExtractor(nn.Module):
    def __init__(self,
                 nclasses):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.feature_dim = self.model.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, nclasses)

    def forward(self, x):
        features = self.model(x) # [B, D, H', W']
        out = F.avg_pool2d(features, (1, 1)) # [B, D, 1, 1]
        out = out.view(-1, self.feature_dim)
        out = self.classifier(out)
        return out

def main():
    model = ImageExtractor(3)
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    model = model.to(device)
    print(model)

if __name__ == "__main__":
    main()
