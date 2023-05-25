"""
Author:Purnasai
Description:This file loads h5py file and 
           searches for match image.
"""
import os
import random
import h5py
import faiss

from PIL import Image
import numpy as np

import torch
from torchvision import transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')


transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])


features_file_path = "features/image_features_dino.h5"
print(f"The Features File size:", round(os.path.getsize(features_file_path)/1000000,2),"MB \n")

# Open the HDF5 file for reading
with h5py.File(features_file_path, 'r') as h5f:
    # Read the dataset named "jewellery_features"
    jewellery_features = np.array(h5f['image_features'])
    # Read the dataset named "jewellery_names"
    jewellery_filenames = np.array(h5f['image_filenames'])


# Print the shape of the arrays to verify the data
print("jewellery_features shape:", type(jewellery_features), jewellery_features.shape)
print("jewellery_names shape:", jewellery_filenames.shape, type(jewellery_filenames))
print("sample:", random.choices(jewellery_filenames,k =5))

# The Inner Product similarity is often used in scenarios 
# where vectors represent semantic or conceptual features.
faiss_index = faiss.IndexFlatIP(jewellery_features.shape[1])
faiss_index.add(jewellery_features)

# L2norm is Euclidean distance, i.e disimilarity. 100-disimarlity
# faiss_index =  faiss.IndexFlatL2(jewellery_features.shape[1])
# faiss_index.add(jewellery_features)

image = Image.open("./Data/Artists/artist1/Pic09345_1.jpeg")
t_image = transform(image).unsqueeze(dim=0)
querry_features = dinov2_vits14(t_image)
querry_features /= querry_features.norm(dim=-1, keepdim=True)
querry_features = querry_features.detach().numpy()

K_neighbours = 10  # number of neighbors to retrieve
distances, indices = faiss_index.search(querry_features, K_neighbours)
for index in range(K_neighbours):
    print(jewellery_filenames[indices[0][index]], distances[0][index]*100)
