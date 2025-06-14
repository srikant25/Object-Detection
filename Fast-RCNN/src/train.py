import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import FastRCNN
from dataset import FastRCNNDataset
from utils import get_target_bbox

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'PennFudanPed/PNGImages')
pkl_path = os.path.join(root_dir, 'Fast-RCNN/data')

# Load region proposals
with open(os.path.join(pkl_path, 'region_proposals.pkl'), 'rb') as f:
    proposal_data = pickle.load(f)

# Hyperparameters
BATCH_SIZE = 2 
NUM_EPOCHS = 70
LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Transforms
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and Loader
dataset = FastRCNNDataset(proposal_data, image_folder, transform=transform)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)

# Model
model = FastRCNN(num_classes=2).to(DEVICE)

# Loss and Optimizer
cls_criterion = nn.CrossEntropyLoss()
reg_criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# LR Shedulingoptimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=3)

best_loss = float('inf')
early_stop_counter = 0
patience = 7

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_cls_loss = 0.0
    total_reg_loss = 0.0

    for batch in train_loader:
        images = []
        rois = []
        labels = []
        target_bboxes = []

        # Process each image in the batch
        for batch_idx, (image_tensor, filtered_proposals) in enumerate(batch):
            image_tensor = image_tensor.to(DEVICE)
            images.append(image_tensor)

            for p in filtered_proposals:
                rois.append([batch_idx] + p['region_proposal_box'])
                labels.append(p['label'])

                if p['label'] == 1:
                    target = get_target_bbox(p['ground_truth_box'], p['region_proposal_box'])
                    target_bboxes.append(target)

        images = torch.stack(images).to(DEVICE)
        rois_tensor = torch.tensor(rois, dtype=torch.float32).to(DEVICE)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(DEVICE)

        # Forward pass
        class_logits, pred_bbox = model(images, rois_tensor)
        cls_loss = cls_criterion(class_logits, labels_tensor)

        # Regression loss for positive samples
        pos_mask = labels_tensor == 1

        if pos_mask.sum() > 0:
            pred_bbox_pos = pred_bbox[pos_mask]
            target_bboxes_tensor = torch.tensor(target_bboxes, dtype=torch.float32).to(DEVICE)
            reg_loss = reg_criterion(pred_bbox_pos, target_bboxes_tensor)
        else:
            reg_loss = torch.tensor(0.0).to(DEVICE)

        # Backpropagation
        loss = cls_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_reg_loss += reg_loss.item()

    epoch_loss = total_cls_loss + total_reg_loss
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - "
          f"Classification Loss: {total_cls_loss:.4f}, "
          f"Regression Loss: {total_reg_loss:.4f}")

    # Step LR scheduler
    scheduler.step(epoch_loss)

    # Early stopping check
    if epoch_loss < best_loss - 1e-4:
        best_loss = epoch_loss
        early_stop_counter = 0

        # Save best model
        save_path = os.path.join("saved_models", "fast_rcnn_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Model improved. Saved to {save_path}")
    else:
        early_stop_counter += 1
        print(f"[INFO] No improvement. Early stop patience: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience: 
        print("[INFO] Early stopping triggered.")
        break

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Classification Loss: {total_cls_loss:.4f}, Regression Loss: {total_reg_loss:.4f}")

# # Save model
# save_path = os.path.join("saved_models", "fast_rcnn.pth")
# os.makedirs(os.path.dirname(save_path), exist_ok=True)
# torch.save(model.state_dict(), save_path)
