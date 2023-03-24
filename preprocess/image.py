import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from multiprocessing import Pool

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Specify the directory where the images are stored
data_dir = 'data/Flicker8k_Dataset'

# Create a new directory to store the processed images
processed_dir = 'data/processed_images'
os.makedirs(processed_dir, exist_ok=True)

# Define a function to load and transform an image
def process_image(image_path):
    image = Image.open(image_path)
    processed_image = transform(image)
    return processed_image

# Get a list of image paths
image_paths = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

# Create a pool of workers to load and transform the images in parallel
with Pool() as pool:
    processed_images = pool.map(process_image, image_paths)

# Convert the list of processed images to a tensor
image_tensor = torch.stack(processed_images, dim=0)

print(image_tensor.size())
torch.save(image_tensor, os.path.join("preprocess/preprocessed", 'images.pt'))