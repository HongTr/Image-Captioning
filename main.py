import argparse
from preprocess.preprocess import handling_token, text_preprocessing, image_processing
from constants import *
from model.model import Model
from train.train import train

parser = argparse.ArgumentParser(description="This is just a description")
parser.add_argument('-m', '--model', action='store', help="model's name", required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-d', '--data', action='store_true', help='data preprocessing')
group.add_argument('-t', '--train', action='store_true', help='train model')
group.add_argument('-e', '--evaluate', action='store_true', help='evalute model')
args = parser.parse_args()

if args.data:
    print("> Processing Data...\n")

    # Create dictionary mapping from image_id to list of descriptions
    print("> Creating vocab...")
    image_id_to_descriptions, vocab = handling_token(
        dir="Flickr8k.token.txt",
    )

    print("> Vocab size: ", vocab.__len__())
    torch.save(vocab, 'preprocess/preprocessed/vocab.pt')

    # Preprocess image_id_to_descriptions's descriptions. Text Preprocessing
    print("> Text Preprocessing...")
    text_preprocessing(
        dict=image_id_to_descriptions,
        vocab=vocab
    )
    # print(image_id_to_descriptions['1000268201_693b08cb0e']) DEBUG
    
    print("> Image Preprocessing...")
    image_processing()

    print("\n> Done!")

if args.train:
    # Load dataset
    print("> Load dataset...\n")

    train_set = open(DATA_DIR + "Flickr_8k.trainImages.txt", 'r').read().splitlines()
    dev_set = open(DATA_DIR + "Flickr_8k.devImages.txt", 'r').read().splitlines()
    vocab = torch.load("preprocess/preprocessed/vocab.pt")
    print("> Train examples: ", len(train_set))
    print("> Dev examples: ", len(dev_set))
    print("> Vocab size:", vocab.__len__())

    print("> Load mapping...")
    image_id_to_image = torch.load('preprocess/preprocessed/image_id_to_image.pt')
    image_id_to_descriptions = torch.load('preprocess/preprocessed/image_id_to_descriptions.pt')

    # Initialize model
    print("> Initialize model...\n")
    model = Model(vocab.__len__()).to(DEVICE)

    # Start training
    print("> Training...\n")
    train(model, train_set, dev_set, image_id_to_image, image_id_to_descriptions)

    print("> Done!\n")

if args.evaluate:
    print("> Evaluating...\n")

    print("> Load dataset...\n")

    print("> Initialize model...\n")

    print("> Load pre-trained model...\n")
        
    # Result