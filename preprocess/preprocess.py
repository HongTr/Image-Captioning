from constants import *
from hyperparameters import *
import string, torch
import torch

from PIL import Image
import os
from multiprocessing import Pool

def handling_token(dir: str) -> dict:
    # A map between image_id and list of descriptions
    mapping = dict()
    # Open file as list of lines
    lines = open(DATA_DIR + dir, 'r').read().splitlines()
    # For each line in lines
    for line in lines:
        # Split line into image_id and description
        image_id, description = line.split(".jpg#")
        description = description[2:]
        # print(image_id, description) DEBUG
        # If image_id is not in map
        if image_id not in mapping:
            # Add image_id to map
            mapping[image_id] = list()
        # Append description to image_id's list of descriptions
        mapping[image_id].append(description)
        # print(mapping[image_id]) DEBUG
    # Return map
    return mapping

def text_preprocessing(dict: dict, output_file_name: str = "token"):
    # Prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # For each key in dictionary
    for key, descriptions in dict.items():
        # For each description in dictionary
        for i in range(len(descriptions)):
            # Create temp string
            temp = descriptions[i]
            # Split into list of word. Tokenize.
            temp = temp.split()
            # Convert to lowercase
            temp = [word.lower() for word in temp]
            # Remove punctuation
            temp = [word.translate(table) for word in temp]
            # Remove 's' and 'a'
            temp = [word for word in temp if len(word) > 1]
            # Remove token with numbers in them
            temp = [word for word in temp if word.isalpha()]
            # Store as list of tokens
            descriptions[i] = temp
    # Save as .pt
    torch.save(dict, f'preprocess/preprocessed/{output_file_name}.pt')

image_text = dict()  #dictionary image name and text
tensor_image = dict() 

def process_image(image_path):
    image = Image.open(image_path)
    processed_image = transform(image)
    return processed_image  

def image_to_text():
    lines = open("data/Flickr8k.token.txt", 'r').read().splitlines()
    global image_text
    for line in lines:
        line = line.split('\t')
        image, text = line[0][:-6], line[1]
        if image not in image_text:
            image_text[image] = list()
            image_text[image].append(text)
        else:
            image_text[image].append(text) 
    torch.save(image_text, os.path.join("preprocess/preprocessed", 'image_text.pt'))

def image_processing(data_dir = IMG_DIR):
    image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

    for index, image in enumerate(image_paths):
        tensor_image[index] = image_paths[index][23:-4]

    with Pool() as pool:
        processed_images = pool.map(process_image,image_paths)
    image_tensor = torch.stack(processed_images, dim=0)
    torch.save(image_tensor, os.path.join("preprocess/preprocessed", 'images.pt'))
    torch.save(tensor_image, os.path.join("preprocess/preprocessed", 'tensor_image.pt'))