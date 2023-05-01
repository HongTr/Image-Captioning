import torch
import torch.nn as nn
from constants import *
from hyperparameters import *
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from evaluation.beam_search import BeamSearch

def evaluate(model: nn.Module, val_set: DataLoader, vocab: Vocab):
    # Initialize some variables
    bleu_per_epoch = 0

    # Make sure gradient tracking is on, and do a pass over the data
    model.eval()

    for data in tqdm(val_set):
        # Extract tensor from dict
        image_tensor = data["image"].to(DEVICE)
        description_tensor = data["description"]

        bleu_per_batch = 0

        # Forward
        output = model(image_tensor, description_tensor)
        output = torch.argmax(output, dim=0)

        for j in range(output.shape[0]):
            translated_output = vocab.lookup_tokens(output[j].cpu().numpy())
            translated_target = [vocab.lookup_tokens(description_tensor[j].cpu().numpy())]
            bleu_per_tensor = sentence_bleu(translated_target, translated_output, weights=(1.0, 0, 0, 0))
            bleu_per_batch += bleu_per_tensor

        bleu_per_batch = bleu_per_batch / BATCH_SIZE
        bleu_per_epoch += bleu_per_batch

    print("> BLEU-1: ", bleu_per_epoch)

def evaluate_beam_seach(model: nn.Module, val_set: DataLoader, vocab: Vocab):
    # Initialize some variables
    bleu_per_epoch = 0

    # Make sure gradient tracking is on, and do a pass over the data
    model.eval()

    for data in tqdm(val_set):
        # Extract tensor from dict
        image_tensor = data["image"].to(DEVICE)
        description_tensor = data["description"]

        bleu_per_batch = 0


        # Initialize the beam search object
        beam_search = BeamSearch(model, BEAM_SIZE, MAX_TGT_SEQ_LENGTH, vocab['<sos>'], vocab['<eos>'], DEVICE)

        # Generate the predicted sequence and compute the BLEU-4 score
        predicted_tokens, sequence_bleu = beam_search.search(image_tensor, vocab['trg'])

        # Update the overall BLEU-4 score
        bleu_score += sequence_bleu

    # Compute the average BLEU-4 score over the validation set
    bleu_score /= len(val_set)

    print("> BLEU-4: ", bleu_per_epoch)