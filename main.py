import argparse
import torch
from preprocess.preprocess_text import handling_corpus

parser = argparse.ArgumentParser(description="This is just a description")
parser.add_argument('-m', '--model', action='store', help="model's name", required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-d', '--data', action='store_true', help='data preprocessing')
group.add_argument('-t', '--train', action='store_true', help='train model')
group.add_argument('-e', '--evaluate', action='store_true', help='evalute model')
args = parser.parse_args()

if args.data:
    print("> Processing Data...\n")

    # Preproccess
    handling_corpus(
        tgt_dir="Flickr8k.token.txt",
        output_file_name="text"
    )

    print("> Done!\n")

if args.train:
    # Load dataset
    print("> Load dataset...\n")

    # Initialize model
    print("> Initialize model...\n")

    # Start training
    print("> Training...\n")

    print("> Done!\n")

if args.evaluate:
    print("> Evaluating...\n")

    print("> Load dataset...\n")

    print("> Initialize model...\n")

    print("> Load pre-trained model...\n")
        
    # Result