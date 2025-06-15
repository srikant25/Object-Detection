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
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.region_proposals)

    def __getitem__(self, idx):
        proposal = self.region_proposals[idx]
        file_name = proposal['file_name']
        image_path = os.path.join(self.image_folder, file_name)

        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # Extract region proposal from image
        x1, y1, x2, y2 = proposal['region_proposal_box']
        roi = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

        label = torch.tensor(proposal['label'], dtype=torch.long)
        target_box = torch.tensor(proposal['target_box'], dtype=torch.float32)

        return image, roi, target_box, label
