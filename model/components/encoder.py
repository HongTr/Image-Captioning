import torch
import torch.nn as nn
from hyperparameters import *
from constants import *

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

    def forward(self, input, target=None):
        pass