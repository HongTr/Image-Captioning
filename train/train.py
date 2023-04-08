import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from hyperparameters import *
from tqdm import tqdm
from utils.utils import model_bleu_score

def train_per_iter(train_set: list, 
        image_id_to_image: dict, 
        image_id_to_description: dict,
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        criterion: nn.NLLLoss
    ):
    # Initialize some variables
    final_loss = 0
    current_loss = 0

    for image_id in tqdm(train_set):
        # Extract tensor from dict
        image_tensor = image_id_to_image[image_id].to(DEVICE)
        description_tensors = image_id_to_description[image_id]

        for tensor in description_tensors:
            # Clear cache
            optimizer.zero_grad()
            
            # Foward
            output = model(image_tensor, tensor)

            # Compute loss
            loss = criterion(output, tensor)

            # Update loss
            current_loss += loss.item()

            # Backward
            loss.backward()

            # Update parameters
            optimizer.step()

    final_loss = current_loss / len(train_set)
    return final_loss

def dev_per_iter(dev_set: list, 
        image_id_to_image: dict, 
        image_id_to_description: dict,
        model: nn.Module, 
        criterion: nn.NLLLoss
    ):
    # Initialize some variables
    final_loss = 0
    current_loss = 0

    for image_id in tqdm(dev_set):
        # Extract tensor from dict
        image_tensor = image_id_to_image[image_id].to(DEVICE)
        description_tensors = image_id_to_description[image_id]

        for tensor in description_tensors:
            
            # Foward
            output = model(image_tensor, tensor)

            # Compute loss
            loss = criterion(output, tensor)

            # Update loss
            current_loss += loss.item()

    final_loss = current_loss / len(dev_set)
    return final_loss

def train(model: nn.Module, 
        train_set: list, 
        dev_set: list, 
        image_id_to_image: dict,
        image_id_to_description: dict,
        vocab
    ):
    # Initialize some variables
    plot_train_loss = []
    plot_dev_loss = []
    plot_train_bleu = []
    plot_dev_bleu = []

    # Loss & Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    # Training
    for epoch in range(EPOCHS):
        # Print epoch/EPOCHS:
        print(f"Epoch: {epoch+1}/{EPOCHS}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)   

        # Train per iteration
        train_average_loss = train_per_iter(train_set, image_id_to_image, image_id_to_description, model, optimizer, criterion)

        # Turn off gradient tracking cause it is not needed anymore
        model.train(False)

        # Calculate Loss on dev set
        dev_average_loss = dev_per_iter(dev_set, image_id_to_image, image_id_to_description, model, criterion)

        # Calculate bleu on train set
        train_bleu = model_bleu_score(train_set, image_id_to_image, image_id_to_description, model, vocab)
        dev_bleu = model_bleu_score(dev_set, image_id_to_image, image_id_to_description, model, vocab)

        # Save loss, bleu
        plot_train_loss.append(train_average_loss)
        plot_dev_loss.append(dev_average_loss)
        plot_train_bleu.append(train_bleu)
        plot_dev_bleu.append(dev_bleu)

        # Print information
        if epoch % PRINT_EVERY == 0:
            print(f"- Loss       | Train: {train_average_loss:.4f} - Dev: {dev_average_loss:.4f}")
            print(f"- Bleu       | Dev: {dev_bleu:.4f}")

    torch.save(plot_train_loss, f'graphs/data/{model.name}_train_loss')
    torch.save(plot_dev_loss, f'graphs/data/{model.name}_dev_loss')
    torch.save(plot_dev_bleu, f'graphs/data/{model.name}_dev_bleu')