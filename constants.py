import torch
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/"
IMG_DIR = "data/Flicker8k_Dataset"

transform = transforms.Compose([
                         transforms.Resize(512),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                         ])