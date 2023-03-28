import torch
import torch.nn as nn
from hyperparameters import *
from constants import *

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=0,
            hidden_size=HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT_RATE,
            bidirectional=False
        )

    def forward(self, input, target=None):
        pass