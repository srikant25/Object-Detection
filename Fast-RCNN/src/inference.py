import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.ops as ops

from model import FastRCNN
from utils import (
    NMS, evaluate_mAP, apply_deltas, get_iou,
    get_target_bbox, visualize,apply_deltas2
)
from data_utils import process_image
import json

import matplotlib.pyplot as plt
import cv2



# ------------------------ Setup ------------------------

# Set paths
root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'PennFudanPed/PNGImages')
bbox_folder = os.path.join(root_dir,'PennFudanPed/boundary_box')
model_path = os.path.join(root_dir, "Fast-RCNN/src/saved_models/fast_rcnn.pth")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ------------------------ Load Model ------------------------

model = FastRCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------------ Load Image & Proposals ------------------------

# Select test image
filename = os.listdir(image_folder)[70]
img_path = os.path.join(image_folder, filename)
annot_file = os.path.join(bbox_folder,filename.replace('.png','.json'))
img = cv2.imread(img_path)
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with open(annot_file, 'r') as f:
    annot = json.load(f)

ground_truth_bboxes = annot['boundary_box'] 
h, w = img.shape[:2]
scale_x = 224 / w
scale_y = 224 / h

# Transform image
image_tensor = transform(image_rgb).unsqueeze(0).to(device)

# Process region proposals
proposals = process_image(filename)
print(f"[INFO] Total proposals: {len(proposals)}")

# Extract ROIs and labels
rois = np.array([p['region_proposal_box'] for p in proposals]).astype(np.float32)
labels = [p['label'] for p in proposals]

# Add batch index to ROIs
if rois.ndim == 2 and rois.shape[1] == 4:
    batch_idx = np.zeros((rois.shape[0], 1), dtype=np.float32)
    rois_arr = np.hstack((batch_idx, rois))
rois_tensor = torch.tensor(rois_arr, dtype=torch.float32).to(device)

# ------------------------ Model Prediction ------------------------

with torch.no_grad():
    probs, deltas = model.predict(image_tensor, rois_tensor)
    probs = probs[:, 1].cpu().numpy()  # Class 1: pedestrian
    deltas = deltas.cpu().numpy()

# ------------------------ Evaluate mAP & IOU ------------------------

# Keep only positive samples for evaluation
positive_indices = [i for i, p in enumerate(proposals) if p['label'] == 1]
target_boxes = [
    get_target_bbox(proposals[i]['ground_truth_box'], proposals[i]['region_proposal_box'])
    for i in positive_indices
]
print(target_boxes)
pred_boxes_eval = deltas[positive_indices]
labels_eval = [labels[i] for i in positive_indices]
print(pred_boxes_eval)

cls_ap, mean_iou = evaluate_mAP(labels_eval, probs[positive_indices], target_boxes, pred_boxes_eval)
print(f"[INFO] Class accuracy: {cls_ap:.3f}")
print(f"[INFO] Mean IOU: {mean_iou:.3f}")

visualize(image_rgb, pred_boxes_eval, target_boxes, label='pedestrian', scores=probs)


# ------------------------ Postprocessing ------------------------

# Apply predicted deltas to proposals
pred_trans_boxes = [apply_deltas2(p['region_proposal_box'], d) for p, d in zip(proposals, deltas)]
boxes_tensor = torch.tensor(pred_trans_boxes, dtype=torch.float32)
scores_tensor = torch.tensor(probs)

# Score threshold
score_thresh = 0.8
mask = scores_tensor > score_thresh
boxes_tensor = boxes_tensor[mask]
scores_tensor = scores_tensor[mask]

# Apply NMS
keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
final_boxes = boxes_tensor[keep].numpy().astype(int)
final_scores = scores_tensor[keep].numpy()
print(final_boxes)
# Inverse scaling: from 224x224 back to original
final_boxes_unscaled = final_boxes.copy().astype(np.float32)
final_boxes_unscaled[:, [0, 2]] /= scale_x  # x1, x2
final_boxes_unscaled[:, [1, 3]] /= scale_y  # y1, y2
final_boxes_unscaled = final_boxes_unscaled.astype(int)


# ------------------------ Ground Truth for Visualization ------------------------

gt_boxes = [p['ground_truth_box'] for p in proposals if p['label'] == 1]
# Inverse scaling: from 224x224 back to original
gt_boxes = np.array(gt_boxes, dtype=np.float32)

# Inverse scaling
gt_boxes[:, [0, 2]] /= scale_x  # x1, x2
gt_boxes[:, [1, 3]] /= scale_y  # y1, y2

# Convert to integer for visualization
gt_boxes_unscaled = gt_boxes.astype(int)
# ------------------------ Visualization ------------------------

visualize(image_rgb, final_boxes_unscaled, gt_boxes_unscaled, label='pedestrian', scores=final_scores)



