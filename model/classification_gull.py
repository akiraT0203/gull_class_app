import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
import pytorch_lightning as pl
from torchvision.models import resnet50

# ResNetに合わせた前処理を追加
def transform(img):
    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return _transform(img)

# ネットワーク
class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 3) # 全結合層で1000から3に繋げる


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h
