import torch
import torch.nn as nn
from hyperparameters import *
from constants import *

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_SIZE
        )

        self.lstm = nn.LSTM(
            input_size=0,
            hidden_size=HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            dropout=DROPOUT_RATE,
            bidirectional=False
        )

    def forward(self, input, decoder_hidden_state, cell_state):
        embedding = self.embedding(input)

        # LSTM
        decoder_hidden_states, (decoder_hidden_state, cell_state) = self.lstm(embedding, (decoder_hidden_state, cell_state))

        return decoder_hidden_states, decoder_hidden_state, cell_state