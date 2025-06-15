# import os
# import pickle
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision.transforms as T
# from model import FastRCNN
# from dataset import FastRCNNDataset
# from utils import get_target_bbox

# # Setup
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# root_dir = os.getcwd()
# proposals_path = os.path.join(root_dir,'Fast-RCNN/data/region_proposals.pkl')
# image_folder = os.path.join(root_dir,'Fast-RCNN/resized_images')




# # Hyperparameters
# BATCH_SIZE = 2 
# NUM_EPOCHS = 70
# LEARNING_RATE = 0.01
# MOMENTUM = 0.9


# # Transforms
# transform = T.Compose([
#     T.ToPILImage(),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# dataset = FastRCNNDataset(proposals_path, image_folder, transform=transform)
# train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)



# # Model
# model = FastRCNN(num_classes=2).to(DEVICE)

# # Loss and Optimizer
# cls_criterion = nn.CrossEntropyLoss()
# reg_criterion = nn.SmoothL1Loss()
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# # LR Shedulingoptimizer
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.1, patience=3)

# best_loss = float('inf')
# early_stop_counter = 0
# patience = 7

# # Training loop
# for epoch in range(NUM_EPOCHS):
#     model.train()
#     total_cls_loss = 0.0
#     total_reg_loss = 0.0

#     for batch in train_loader:
#         images = []
#         rois = []
#         labels = []
#         target_bboxes = []

#         # Process each image in the batch
#         for batch_idx, (image_tensor, filtered_proposals) in enumerate(batch):
#             image_tensor = image_tensor.to(DEVICE)
#             images.append(image_tensor)

#             for p in filtered_proposals:
#                 rois.append([batch_idx] + p['region_proposal_box'])
#                 labels.append(p['label'])

#                 if p['label'] == 1:
#                     target = get_target_bbox(p['ground_truth_box'], p['region_proposal_box'])
#                     target_bboxes.append(target)

#         images = torch.stack(images).to(DEVICE)
#         rois_tensor = torch.tensor(rois, dtype=torch.float32).to(DEVICE)
#         labels_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)

#         # Forward pass
#         class_logits, pred_bbox = model(images, rois_tensor)
#         cls_loss = cls_criterion(class_logits, labels_tensor)

#         # Regression loss for positive samples
#         pos_mask = labels_tensor == 1

#         if pos_mask.sum() > 0:
#             pred_bbox_pos = pred_bbox[pos_mask]
#             target_bboxes_tensor = torch.tensor(target_bboxes, dtype=torch.float32).to(DEVICE)
#             reg_loss = reg_criterion(pred_bbox_pos, target_bboxes_tensor)
#         else:
#             reg_loss = torch.tensor(0.0).to(DEVICE)

#         # Backpropagation
#         loss = cls_loss + reg_loss
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
#         optimizer.step()

#         total_cls_loss += cls_loss.item()
#         total_reg_loss += reg_loss.item()

#     epoch_loss = total_cls_loss + total_reg_loss
#     print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - "
#           f"Classification Loss: {total_cls_loss:.4f}, "
#           f"Regression Loss: {total_reg_loss:.4f}")

#     # Step LR scheduler
#     scheduler.step(epoch_loss)

#     # Early stopping check
#     if epoch_loss < best_loss - 1e-4:
#         best_loss = epoch_loss
#         early_stop_counter = 0

#         # Save best model
#         save_path = os.path.join("saved_models", "fast_rcnn_best.pth")
#         torch.save(model.state_dict(), save_path)
#         print(f"[INFO] Model improved. Saved to {save_path}")
#     else:
#         early_stop_counter += 1
#         print(f"[INFO] No improvement. Early stop patience: {early_stop_counter}/{patience}")

#     if early_stop_counter >= patience: 
#         print("[INFO] Early stopping triggered.")
#         break

#     print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Classification Loss: {total_cls_loss:.4f}, Regression Loss: {total_reg_loss:.4f}")

# # # Save model
# # save_path = os.path.join("saved_models", "fast_rcnn.pth")
# # os.makedirs(os.path.dirname(save_path), exist_ok=True)
# # torch.save(model.state_dict(), save_path)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
from model import FastRCNN
from dataset import FastRCNNDataset

# === Configurations ===
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'Fast-RCNN/data')
IMAGE_DIR = os.path.join(BASE_DIR, 'Fast-RCNN/resized_images')
PROPOSALS_PATH = os.path.join(DATA_DIR, 'region_proposals.pkl')
SAVE_MODEL_PATH = os.path.join(BASE_DIR, 'Fast-RCNN/src/saved_models')
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

# === Transform ===
# Transforms
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Dataset and DataLoader ===
dataset = FastRCNNDataset(proposals_path=PROPOSALS_PATH, image_folder=IMAGE_DIR, transform=transform)

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def collate_fn(batch):
    """
    Custom collate function to pad images in the batch and prepare RoIs with batch indices.
    """
    images, rois, target_boxes, labels = zip(*batch)

    # Find max height and width
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    roi_batch = []
    for i, img in enumerate(images):
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w

        # Pad image on bottom and right
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        padded_images.append(padded_img)

        # roi[i] is shape [4] — convert to shape [1, 4]
        roi = rois[i].unsqueeze(0)  # [1, 4]
        batch_idx = torch.full((1, 1), i, dtype=torch.float32)  # [[i]]
        roi_with_idx = torch.cat([batch_idx, roi], dim=1)  # [1, 5]
        roi_batch.append(roi_with_idx)

    batch_images = torch.stack(padded_images)  # [B, 3, H, W]
    batch_rois = torch.cat(roi_batch, dim=0)   # [N, 5]
    batch_target_boxes = torch.stack(target_boxes)  # [N, 4]
    batch_labels = torch.stack(labels)              # [N]

    return batch_images, batch_rois, batch_target_boxes, batch_labels


dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# === Model ===
model = FastRCNN(num_classes=2).to(DEVICE)

# === Losses and Optimizer ===
cls_criterion = nn.CrossEntropyLoss()
reg_criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training Loop ===
for epoch in range(NUM_EPOCHS):
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for images, rois, targets, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images = images.to(DEVICE)
        rois = rois.to(DEVICE)
        targets = targets.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        class_logits, bbox_preds = model(images, rois)

        cls_loss = cls_criterion(class_logits, labels)
        
        # Apply regression loss only on positive samples
        pos_indices = labels == 1
        if pos_indices.sum() > 0:
            reg_loss = reg_criterion(bbox_preds[pos_indices], targets[pos_indices])
        else:
            reg_loss = torch.tensor(0.0, device=DEVICE)

        loss = cls_loss + reg_loss
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()

    print(f"[Epoch {epoch+1}] Classification Loss: {total_cls_loss:.4f} | Regression Loss: {total_reg_loss:.4f}")

    # Save model after each epoch
    model_path = os.path.join(SAVE_MODEL_PATH, f"fastrcnn_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)

print("Training Complete!")

