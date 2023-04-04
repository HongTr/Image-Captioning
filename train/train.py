import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from hyperparameters import *
from timeit import default_timer as timer

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

    for image_id in train_set:
        # Extract tensor from dict
        image_tensor = image_id_to_image[image_id]
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

def train(model: nn.Module, 
        train_set: list, 
        dev_set: list, 
        image_id_to_image: dict, 
        image_id_to_description: dict
    ):
    # Initialize some variables

    # Loss & Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()

    # Training
    for epoch in range(EPOCHS):
        # Start timer
        start_time = timer()

        # Print epoch/EPOCHS:
        print(f"Epoch: {epoch+1}/{EPOCHS}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)   

        # Train per iteration
        train_average_loss = train_per_iter(train_set, image_id_to_image, image_id_to_description, model, optimizer, criterion)

        # Turn off gradient tracking cause it is not needed anymore
        model.train(False)

        # Calculate Loss on dev set

        # Calculate bleu on train set

        # End timer
        end_time = timer()

        # Print information
        if epoch % PRINT_EVERY == 0:
            print("Done")