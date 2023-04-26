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
            input_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            bidirectional=False,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.linear = nn.Linear(
            in_features=HIDDEN_SIZE,
            out_features=vocab_size
        )
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, decoder_hidden_state, cell_state):
        embedding = self.dropout(self.embedding(input))

        # LSTM
        decoder_hidden_states, (decoder_hidden_state, cell_state) = self.lstm(embedding, (decoder_hidden_state, cell_state))

        output = self.log_softmax(self.linear(decoder_hidden_states))

        return output, decoder_hidden_state, cell_state