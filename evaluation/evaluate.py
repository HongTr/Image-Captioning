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
        output = torch.argmax(output, dim=1)

        for j in range(BATCH_SIZE):
            translated_output = vocab.lookup_tokens(output.cpu().numpy())
            translated_target = [vocab.lookup_tokens(description_tensor.cpu().numpy())]
            bleu_per_tensor = sentence_bleu(translated_target, translated_output, weights=(1.0, 0, 0, 0))
            bleu_per_batch += bleu_per_tensor

        bleu_per_batch = bleu_per_batch / BATCH_SIZE
        bleu_per_epoch += bleu_per_batch

    print("> BLEU-1: ", bleu_per_epoch)