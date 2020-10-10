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
        self.classifier = nn.Linear(self.model.fc.in_features, nclasses)

    def forward(self, x):
        features = self.model(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=2).view(features.size(0), -1)
        out = F.sigmoid(self.classifier(out))
        return out

def main():
    model = ImageExtractor(3)
    dev_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    model = model.to(device)
    print(model)

if __name__ == "__main__":
    main()
