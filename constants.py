import torch

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/"
IMG_DIR = "data/Flicker8k_Dataset"