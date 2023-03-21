import argparse
from preprocess.preprocess import handling_token, text_preprocessing

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
    image_id_to_descriptions = handling_token(
        dir="Flickr8k.token.txt",
    )

    # Preprocess image_id_to_descriptions's descriptions. Text Preprocessing
    text_preprocessing(
        dict=image_id_to_descriptions
    )
    # print(image_id_to_descriptions['1000268201_693b08cb0e']) DEBUG

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