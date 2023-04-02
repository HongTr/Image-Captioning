import torch
import torch.nn as nn
from hyperparameters import *
from constants import *
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, hiddensize):
        super(Encoder, self).__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.aux_logits = False
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, hiddensize),
        )
    def forward(self, input, target=None):
        input = input.unsqueeze(0)
        output = self.model(input)
        return output