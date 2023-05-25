"""
Author:Purnasai
Description:This file generates image features from
        Database of images & stores them h5py file.
"""
import os
import random

import h5py
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

def get_labels(files):
    """
    This function takes a list of file paths and returns a list of unique labels extracted from the
    directory names in the file paths.
    
    :param files: a list of file paths (strings) that include the directory and filename, separated
    by backslashes ("\") on Windows or forward slashes ("/") on Unix-based systems
    :return: a list of unique labels extracted from the file paths provided in the `files` parameter.
    """
    labels = []
    for file_path in files:
        directory, _ = file_path.split("\\")
        directory_parts = directory.split("/")
        label = directory_parts[-1]
        if label not in labels:
            labels.append(label)
    return labels

def list_files(dataset_path):
    """
    This function returns a list of all files in a directory and its subdirectories.
    
    :param dir: The directory path where you want to list all the files
    :return: The function `list_files` returns a list of file paths for all the files in the directory
    and its subdirectories.
    """
    images = []
    for root, _, files in os.walk(dataset_path):
        for name in files:
            images.append(os.path.join(root, name))
    return images


class CustomImageDataset(Dataset):
    """The above class is a custom dataset class for images in PyTorch."""
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list_files(self.img_dir)
        random.choices(self.images, k=5)
        self.transform =  transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


dir_path = "./Data/"
dataset = CustomImageDataset(dir_path)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True,)


final_img_features = []
final_img_filepaths = []
for image_tensors, file_paths in tqdm(train_dataloader):
    try:
        image_features = dinov2_vitl14(image_tensors) #384 small, #768 base, #1024 large
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.tolist()
        final_img_features.extend(image_features)
        final_img_filepaths.extend((list(file_paths)))
    except Exception as e:
        print("Exception occurred: ",e)
        break


with h5py.File('features/image_features_dino.h5','w') as h5f:
    h5f.create_dataset("image_features", data= np.array(final_img_features))
    # to save file names strings in byte format.
    h5f.create_dataset("image_filenames", data= np.array(final_img_filepaths,
                                                             dtype=object))
