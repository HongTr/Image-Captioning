import argparse
from preprocess.preprocess import handling_token, text_preprocessing, image_processing, create_dataloader, process_image
from constants import *
from hyperparameters import *
from model.model import Model
from train.train import train
import os
from evaluation.evaluate import evaluate
from urllib.request import urlopen

parser = argparse.ArgumentParser(description="This is just a description")
parser.add_argument('-m', '--model', action='store', help="model's name", required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-d', '--data', action='store_true', help='data preprocessing')
group.add_argument('-t', '--train', action='store_true', help='train model')
group.add_argument('-e', '--evaluate', action='store_true', help='evalute model')
group.add_argument('-tr', '--translate', action='store_true', help='translate through url')
args = parser.parse_args()

if args.data:
    print("> Processing Data...\n")

    # Create dictionary mapping from image_id to list of descriptions
    print("> Creating vocab...")
    image_id_to_descriptions, vocab = handling_token(
        dir="Flickr8k.token.txt",
    )

    print("> Vocab size: ", vocab.__len__())
    if os.path.isdir('preprocess/preprocessed/') is False:
        os.makedirs('preprocess/preprocessed/')
    torch.save(vocab, 'preprocess/preprocessed/vocab.pt')

    # Preprocess image_id_to_descriptions's descriptions. Text Preprocessing
    print("> Text Preprocessing...")
    text_preprocessing(
        dict=image_id_to_descriptions,
        vocab=vocab
    )
    
    print("> Image Preprocessing...")
    image_processing()

    print("\n> Done!")

if args.train:
    # Load dataset
    print("> Load dataset...")

    train_set = open(DATA_DIR + "Flickr_8k.trainImages.txt", 'r').read().splitlines()
    vocab = torch.load("preprocess/preprocessed/vocab.pt")
    print("> Train examples: ", len(train_set))
    print("> Vocab size:", vocab.__len__())

    print("> Load mapping...")
    image_id_to_image = torch.load('preprocess/preprocessed/image_id_to_image.pt')
    image_id_to_descriptions = torch.load('preprocess/preprocessed/image_id_to_descriptions.pt')

    print("> Initialize DataLoader...")
    train_set = create_dataloader(train_set, image_id_to_image, image_id_to_descriptions)

    # Initialize model
    print("> Initialize model...")
    model = Model(vocab.__len__()).to(DEVICE)

    # Start training
    print("> Training...")
    train(model, train_set)

    print("> Done!\n")

if args.evaluate:
    print("> Evaluating...")

    print("> Load dataset...")
    val_set = open(DATA_DIR + "Flickr_8k.devImages.txt", 'r').read().splitlines()
    vocab = torch.load("preprocess/preprocessed/vocab.pt")
    print("> Val examples: ", len(val_set))
    print("> Vocab size:", vocab.__len__())
    
    print("> Load mapping...")
    image_id_to_image = torch.load('preprocess/preprocessed/image_id_to_image.pt')
    image_id_to_descriptions = torch.load('preprocess/preprocessed/image_id_to_descriptions.pt')

    print("> Initialize DataLoader...")
    val_set = create_dataloader(val_set, image_id_to_image, image_id_to_descriptions)

    print("> Initialize model...")
    model = Model(vocab.__len__()).to(DEVICE)

    print("> Load pre-trained model...")
    state_dict = torch.load("model/snapshot/20230508_082303/snap_shot_1.pt", map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)

    evaluate(model, val_set, vocab)

if args.translate:
    print("> Translation")

    print("> Load vocab...")
    vocab = torch.load("preprocess/preprocessed/vocab.pt")
    print("> Vocab size:", vocab.__len__())

    print("> Initialize model...")
    model = Model(vocab.__len__()).to(DEVICE)

    print("> Load pre-trained model...")
    state_dict = torch.load("model/snapshot/best_model.pt", map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)

    # Load the image from the provided link
    url = input("> Enter image link: ")
    image = process_image(urlopen(url)).unsqueeze(0).to(DEVICE)

    # Put the tensor through the model and get the predicted caption
    with torch.no_grad():
        output = model(image)
        output = torch.argmax(output, dim=0)

        for j in range(output.shape[0]):
            translated_output = vocab.lookup_tokens(output[j].cpu().numpy())

    print("> Caption: ", translated_output)