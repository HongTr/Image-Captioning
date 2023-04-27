import torch
import torch.nn as nn
from hyperparameters import *
from constants import *
from model.components.encoder import Encoder
from model.components.decoder import Decoder
from train.teacher_forcing import teacher_forcing

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
            target_sequence_length = target.shape[1]
        
        # Get Batch_size
        batch_size = input.shape[0]

        outputs = torch.zeros((target_sequence_length, batch_size, self.vocab_size), device=DEVICE)

        # Encoder
        embedding_vector = self.encoder(input)

        # Bridge
        decoder_input = torch.full((batch_size, 1), 2, device=DEVICE)
        decoder_hidden = embedding_vector.unsqueeze(0)
        decoder_cell = torch.zeros(decoder_hidden.shape, device=DEVICE)

        # Decoder
        ## Loop through target
        for i in range(target_sequence_length):
            ## Forward through Decoder
            output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            ## Get next input
            if target is None:
                decoder_input = output.argmax(2)
            else:
                decoder_input = teacher_forcing(output, target[:, i])
            ## Save output for compute loss
            outputs[i] = output.squeeze()

        return outputs