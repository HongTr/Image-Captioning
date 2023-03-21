from constants import *
from hyperparameters import *
import json
import re

def handling_corpus(tgt_dir: str, output_file_name: str) -> None:
    picture_names = []
    descriptions = []
    ## Read dataset .txt to list of one-line code
    tgt = open(DATA_DIR + tgt_dir, 'r').read().splitlines()
    for line in tgt:
        picture_name, description = line.split(".jpg#")
        description = description[2:]
        # print(picture_name, description)
        picture_names.append(picture_name)
        descriptions.append(description)
    src_tgt = list(zip(picture_names, descriptions))
    ## From lists of (source, target) to lists of jsons
    raw_data = [
        {
            "pic": example[0],
            "description": example[1]
        } for example in src_tgt
    ]
    ## From lists of jsons to .json
    with open(f'preprocess/preprocessed/{output_file_name}.json', 'w') as file:
        for item in raw_data:
            json.dump(item, file)
            file.write('\n')