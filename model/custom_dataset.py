import torch
from torch.utils.data import Dataset, DataLoader

class ImageDescriptionDataset(Dataset):
    def __init__(self, images_dict, descriptions_dict):
        self.image_ids = list(images_dict.keys())
        self.images = list(images_dict.values())
        self.descriptions = [desc for sublist in list(descriptions_dict.values()) for desc in sublist]

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx % 5]
        image = self.images[idx % 5]
        description = self.descriptions[idx]
        return {'image_id': image_id, 'image': image, 'description': description}