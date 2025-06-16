import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
import torchvision.transforms as T


import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import cv2
import numpy as np
import torchvision.transforms as T

class FastRCNNDataset(Dataset):
    def __init__(self, proposals_path, image_folder, transform=None):
        with open(proposals_path, 'rb') as f:
            self.region_proposals = pickle.load(f)
        self.img_name =[]
        for file in os.listdir(image_folder):
            self.img_name.append(file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, idx):
        file_name = self.img_name[idx]
        proposals = [p for p in self.region_proposals if p['file_name']==file_name]
        image_path = os.path.join(self.image_folder, file_name)

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, proposals

import tensorflow as tf
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
import sys
print(sys.executable)