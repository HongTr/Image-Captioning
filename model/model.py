import torch
import torch.nn as nn
from hyperparameters import *
from constants import *
from components.encoder import Encoder
from components.decoder import Decoder

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.encoder = Encoder(hidden_size)
        self.decoder = Decoder()

    def forward(self, input, target=None):
        pass