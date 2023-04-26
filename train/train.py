import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from hyperparameters import *
from tqdm import tqdm
import os
from datetime import datetime
from torch.utils.data import DataLoader

def train_per_iter(train_set: DataLoader, 
        model: nn.Module, 
        optimizer: optim.Optimizer, 
        criterion: nn.NLLLoss
    ):
    # Initialize some variables
    final_loss = 0
    current_loss = 0

    for data in tqdm(train_set):
        # Extract tensor from dict
        image_tensor = data["image"].to(DEVICE)
        description_tensor = data["description"]

        # Clear cache
        optimizer.zero_grad()
        
        # Foward
        output = model(image_tensor, description_tensor)

        # Compute loss
        loss = criterion(output.view(-1, output.shape[2]), description_tensor.view(description_tensor.shape[0] * description_tensor.shape[1]))

        # Update loss
        current_loss += loss.item()

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()

    final_loss = current_loss / len(train_set)
    return final_loss

def train(model: nn.Module, train_set: DataLoader):
    # Initialize some variables
    plot_train_loss = []
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

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
        train_average_loss = train_per_iter(train_set, model, optimizer, criterion)

        # Turn off gradient tracking cause it is not needed anymore
        model.train(False)

        # Save loss
        plot_train_loss.append(train_average_loss)

        # Print information
        if epoch % PRINT_EVERY == 0:
            if os.path.isdir(f'model/snapshot/{time_stamp}') is False:
                os.makedirs(f'model/snapshot/{time_stamp}')
            torch.save(model.state_dict(), f'model/snapshot/{time_stamp}/snap_shot_{epoch}.pt')
            print(f"- Loss: {train_average_loss:.4f}")

    torch.save(plot_train_loss, f'graphs/data/train_loss')