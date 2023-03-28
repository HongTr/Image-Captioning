from constants import *
from hyperparameters import *
import string, torch
from torchtext.vocab import GloVe

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
    # GloVe vocab
    vocab = GloVe("6B", 200)
    # Prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # For each key in dictionary
    for key, descriptions in dict.items():
        print(key)
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
            # To Vector
            temp = vocab.get_vecs_by_tokens(temp)
            # To tensor
            temp = torch.tensor(temp, dtype=torch.float32, device=DEVICE)
            # Store as list of tokens
            descriptions[i] = temp
    # Save as .pt
    torch.save(dict, f'preprocess/preprocessed/{output_file_name}.pt')
