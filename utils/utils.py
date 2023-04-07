from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperparameters import *
from constants import *
from torchtext.vocab import Vocab
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def model_bleu_score(dataset, image_id_to_image: dict, image_id_to_description: dict, model, vocab: Vocab):
    bleu_per_epoch = 0

    for image_id in tqdm(dataset):
        # Extract tensor from dict
        image_tensor = image_id_to_image[image_id].to(DEVICE)
        description_tensors = image_id_to_description[image_id]\

        for tensor in description_tensors:
            bleu_per_batch = 0

            # Forward
            output = model(image_tensor, tensor)
            output = torch.argmax(output, dim=1)

            # From Tensors to Sentences -> Calculate Bleu on sentence
            for j in range(BATCH_SIZE):
                translated_output = vocab.lookup_tokens(output.cpu().numpy())
                translated_target = vocab.lookup_tokens(tensor.cpu().numpy())
                bleu_per_tensor = sentence_bleu(translated_target, translated_output, weights=(1.0, 0, 0, 0))
                bleu_per_batch += bleu_per_tensor

            bleu_per_batch = bleu_per_batch / BATCH_SIZE
            bleu_per_epoch += bleu_per_batch

    return bleu_per_epoch / len(dataset)

def plot_loss(model):
    plot_train_loss = torch.load(f'graphs/data/{model.name}_train_loss')
    train_min_value = min(plot_train_loss)
    train_min_index = plot_train_loss.index(train_min_value)

    plot_dev_loss = torch.load(f'graphs/data/{model.name}_dev_loss')
    dev_min_value = min(plot_dev_loss)
    dev_min_index = plot_dev_loss.index(dev_min_value)

    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Average train/dev loss per epoch')
    plt.plot(plot_train_loss, 'r', label='Train')
    plt.plot(train_min_index, train_min_value, 'ro')
    plt.annotate(f"{train_min_value:.4f}", (train_min_index, train_min_value), verticalalignment='top')

    plt.plot(plot_dev_loss, 'b', label='Dev')
    plt.plot(dev_min_index, dev_min_value, 'bo')
    plt.annotate(f"{dev_min_value:.4f}", (dev_min_index, dev_min_value), verticalalignment='top')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Average loss')
    plt.legend()
    plt.xticks(np.arange(1, len(plot_train_loss)+1, 5))
    plt.savefig(f"graphs/graphs/loss_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_bleu(model):
    plot_train_bleu = torch.load(f'graphs/data/{model.name}_train_bleu')
    train_max_value = max(plot_train_bleu)
    train_max_index = plot_train_bleu.index(train_max_value)

    plot_dev_bleu = torch.load(f'graphs/data/{model.name}_dev_bleu')
    dev_max_value = max(plot_dev_bleu)
    dev_max_index = plot_dev_bleu.index(dev_max_value)

    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Train/Dev bleu score per epoch')
    plt.plot(plot_train_bleu, 'r', label='Train')
    plt.plot(train_max_index, train_max_value, 'ro')
    plt.annotate(f"{train_max_value:.4f}", (train_max_index, train_max_value))

    plt.plot(plot_dev_bleu, 'b', label='Dev')
    plt.plot(dev_max_index, dev_max_value, 'bo')
    plt.annotate(f"{dev_max_value:.4f}", (dev_max_index, dev_max_value))

    plt.xlabel('Number of Epochs')
    plt.ylabel('Bleu Score')
    plt.legend()
    plt.xticks(np.arange(1, len(plot_train_bleu)+1, 5))
    plt.savefig(f"graphs/graphs/bleu_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_bleu_per_sentence(model):
    bleu_per_sentence = torch.load(f'graphs/data/bleu_per_sentences_{model.name}')
    input_lengths = torch.load(f"graphs/data/inputs_length")
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    plot_bleu_per_sentence = []
    plot_input_lengths = []

    for i in range(len(input_lengths)):
        if i % 55 == 0:
            plot_bleu_per_sentence.append(bleu_per_sentence[i])
            plot_input_lengths.append(input_lengths[i])

    index = np.argsort(plot_input_lengths)
    plot_bleu_per_sentence = np.array(plot_bleu_per_sentence)[index]
    plot_input_lengths = np.sort(plot_input_lengths)

    plt.figure(figsize=(10, 6))
    plt.title(f'{model.name}: Bleu score on each sentence, length increase')
    plt.bar(range(len(plot_bleu_per_sentence)), plot_bleu_per_sentence)
    plt.xticks(range(len(plot_bleu_per_sentence)), plot_input_lengths)
    plt.xlabel("Sentence's length")
    plt.ylabel('Bleu Score')

    plt.savefig(f"graphs/graphs/bleu_per_sentence_{model.name}_{time_stamp}.png")
    plt.figure().clear()