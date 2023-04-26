import torch
import torch.nn as nn
from hyperparameters import *
from constants import *
from torchvision import models
from torchvision.models import Inception_V3_Weights

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.model.aux_logits = False
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, HIDDEN_SIZE),
        )
        self.interpolate = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    def forward(self, input, target=None):
        input = self.interpolate(input)
        target = self.model(input)
        return target