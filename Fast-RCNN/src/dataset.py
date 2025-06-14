import cv2
import os
from torch.utils.data import Dataset
import torchvision.transforms as T


class FastRCNNDataset(Dataset):
    def __init__(self, proposals, image_folder, transform=None):
        self.proposals = proposals
        self.image_folder = image_folder
        self.transform = transform if transform else T.ToTensor()

        # Collect unique image filenames from proposals
        self.image_filenames = list(set([p['file_name'] for p in self.proposals]))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_folder, img_name)

        # Load and convert image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformation (e.g., ToTensor, Normalize, etc.)
        image_tensor = self.transform(image)

        # Get region proposals for this image
        filtered_proposals = [p for p in self.proposals if p['file_name'] == img_name]

        return image_tensor, filtered_proposals



        
