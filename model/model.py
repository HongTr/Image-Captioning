import torch
import torch.nn as nn
from hyperparameters import *
from constants import *
from model.components.encoder import Encoder
from model.components.decoder import Decoder

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size)

    def forward(self, input, target=None):
        # Get target length
        if target == None:
            target_sequence_length = MAX_TGT_SEQ_LENGTH
        else:
            target_sequence_length = target.shape[0]

        outputs = torch.zeros((target_sequence_length, self.vocab_size), device=DEVICE)

        embedding_vector = self.encoder(input)

        decoder_input = torch.full((1), 2, device=DEVICE)
        decoder_hidden = embedding_vector
        decoder_cell = torch.zeros(decoder_hidden.shape, device=DEVICE)

        ## Loop through target
        for i in range(target_sequence_length):
            ## Forward through Decoder
            output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            ## Get next input
            decoder_input = output.argmax(2)
            ## Save output for compute loss
            outputs[i] = output.squeeze()

        return outputs