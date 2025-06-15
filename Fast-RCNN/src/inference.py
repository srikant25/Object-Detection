# import os
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as T
# import torchvision.ops as ops

# from model import FastRCNN
# from utils import (
#     NMS, evaluate_mAP, apply_deltas, get_iou,
#     get_target_bbox, visualize,apply_deltas2
# )
# from data_utils import process_image
# import json
# import cv2



# # ------------------------ Setup ------------------------

# # Set paths
# root_dir = os.getcwd()
# image_folder = os.path.join(root_dir, 'PennFudanPed/PNGImages')
# bbox_folder = os.path.join(root_dir,'PennFudanPed/boundary_box')
# model_path = os.path.join(root_dir, "Fast-RCNN/src/saved_models/fast_rcnn.pth")

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Transforms
# transform = T.Compose([
#     T.ToPILImage(),
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])

# # ------------------------ Load Model ------------------------

# model = FastRCNN()
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()

# # ------------------------ Load Image & Proposals ------------------------

# # Select test image
# filename = os.listdir(image_folder)[70]
# img_path = os.path.join(image_folder, filename)
# annot_file = os.path.join(bbox_folder,filename.replace('.png','.json'))
# img = cv2.imread(img_path)
# image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# with open(annot_file, 'r') as f:
#     annot = json.load(f)

# ground_truth_bboxes = annot['boundary_box'] 
# h, w = img.shape[:2]
# scale_x = 224 / w
# scale_y = 224 / h

# # Transform image
# image_tensor = transform(image_rgb).unsqueeze(0).to(device)

# # Process region proposals
# proposals = process_image(filename)
# print(f"[INFO] Total proposals: {len(proposals)}")

# # Extract ROIs and labels
# rois = np.array([p['region_proposal_box'] for p in proposals]).astype(np.float32)
# labels = [p['label'] for p in proposals]

# # Add batch index to ROIs
# if rois.ndim == 2 and rois.shape[1] == 4:
#     batch_idx = np.zeros((rois.shape[0], 1), dtype=np.float32)
#     rois_arr = np.hstack((batch_idx, rois))
# rois_tensor = torch.tensor(rois_arr, dtype=torch.float32).to(device)

# # ------------------------ Model Prediction ------------------------

# with torch.no_grad():
#     probs, deltas = model.predict(image_tensor, rois_tensor)
#     probs = probs[:, 1].cpu().numpy()  # Class 1: pedestrian
#     deltas = deltas.cpu().numpy()

# # ------------------------ Evaluate mAP & IOU ------------------------

# # Keep only positive samples for evaluation
# positive_indices = [i for i, p in enumerate(proposals) if p['label'] == 1]
# target_boxes = [
#     get_target_bbox(proposals[i]['ground_truth_box'], proposals[i]['region_proposal_box'])
#     for i in positive_indices
# ]
# print(target_boxes)
# pred_boxes_eval = deltas[positive_indices]
# labels_eval = [labels[i] for i in positive_indices]
# print(pred_boxes_eval)

# cls_ap, mean_iou = evaluate_mAP(labels_eval, probs[positive_indices], target_boxes, pred_boxes_eval)
# print(f"[INFO] Class accuracy: {cls_ap:.3f}")
# print(f"[INFO] Mean IOU: {mean_iou:.3f}")

# visualize(image_rgb, pred_boxes_eval, target_boxes, label='pedestrian', scores=probs)


# # ------------------------ Postprocessing ------------------------

# # Apply predicted deltas to proposals
# pred_trans_boxes = [apply_deltas2(p['region_proposal_box'], d) for p, d in zip(proposals, deltas)]
# boxes_tensor = torch.tensor(pred_trans_boxes, dtype=torch.float32)
# scores_tensor = torch.tensor(probs)

# # Score threshold
# score_thresh = 0.8
# mask = scores_tensor > score_thresh
# boxes_tensor = boxes_tensor[mask]
# scores_tensor = scores_tensor[mask]

# # Apply NMS
# keep = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.3)
# final_boxes = boxes_tensor[keep].numpy().astype(int)
# final_scores = scores_tensor[keep].numpy()
# print(final_boxes)
# # Inverse scaling: from 224x224 back to original
# final_boxes_unscaled = final_boxes.copy().astype(np.float32)
# final_boxes_unscaled[:, [0, 2]] /= scale_x  # x1, x2
# final_boxes_unscaled[:, [1, 3]] /= scale_y  # y1, y2
# final_boxes_unscaled = final_boxes_unscaled.astype(int)


# # ------------------------ Ground Truth for Visualization ------------------------

# gt_boxes = [p['ground_truth_box'] for p in proposals if p['label'] == 1]
# # Inverse scaling: from 224x224 back to original
# gt_boxes = np.array(gt_boxes, dtype=np.float32)

# # Inverse scaling
# gt_boxes[:, [0, 2]] /= scale_x  # x1, x2
# gt_boxes[:, [1, 3]] /= scale_y  # y1, y2

# # Convert to integer for visualization
# gt_boxes_unscaled = gt_boxes.astype(int)
# # ------------------------ Visualization ------------------------

# visualize(image_rgb, final_boxes_unscaled, gt_boxes_unscaled, label='pedestrian', scores=final_scores)


import os
import cv2
import torch
import pickle
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import FastRCNN  # your model class

# -------- CONFIG --------
PROPOSAL_PATH = 'Fast-RCNN/data/region_proposals.pkl'
IMAGE_FOLDER = 'PennFudanPed/PNGImages'
MODEL_PATH = 'Fast-RCNN/src/saved_models/fastrcnn.pth'
IMAGE_IDX = 0  # Change index to test different images
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- Load model --------
model = FastRCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- Load region proposals --------
with open(PROPOSAL_PATH, 'rb') as f:
    proposals = pickle.load(f)

# Group proposals by image
from collections import defaultdict
grouped = defaultdict(list)
for p in proposals:
    grouped[p['file_name']].append(p)

# Pick one image
filename = list(grouped.keys())[IMAGE_IDX]
image_path = os.path.join(IMAGE_FOLDER, filename)
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess
transform = transforms.Compose([
    transforms.ToTensor()
])
image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

# Gather RoIs
samples = grouped[filename]
rois = []
gt_boxes = []

for i, sample in enumerate(samples):
    x1, y1, x2, y2 = sample['region_proposal_box']
    rois.append([0, x1, y1, x2, y2])  # batch index = 0
    gt_boxes.append(sample['ground_truth_box'])

rois = torch.tensor(rois, dtype=torch.float32).to(DEVICE)

# -------- Inference --------
with torch.no_grad():
    probs, bbox_deltas = model.predict(image_tensor, rois)

# Decode results
probs = probs.cpu().numpy()
bbox_deltas = bbox_deltas.cpu().numpy()
rois_np = rois.cpu().numpy()

# -------- Visualization --------
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image_rgb)

# Draw ground truth boxes (green)
for box in gt_boxes:
    x1, y1, x2, y2 = map(int, box)
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

# Draw predicted boxes (red)
for i in range(len(rois_np)):
    class_id = np.argmax(probs[i])
    confidence = probs[i][class_id]

    if class_id == 1 and confidence > CONFIDENCE_THRESHOLD:
        # Decode bbox delta from roi
        x1, y1, x2, y2 = rois_np[i][1:]
        dx, dy, dw, dh = bbox_deltas[i]

        width = x2 - x1
        height = y2 - y1
        ctr_x = x1 + 0.5 * width
        ctr_y = y1 + 0.5 * height

        pred_ctr_x = ctr_x + dx * width
        pred_ctr_y = ctr_y + dy * height
        pred_w = np.exp(dw) * width
        pred_h = np.exp(dh) * height

        pred_x1 = int(pred_ctr_x - 0.5 * pred_w)
        pred_y1 = int(pred_ctr_y - 0.5 * pred_h)
        pred_x2 = int(pred_ctr_x + 0.5 * pred_w)
        pred_y2 = int(pred_ctr_y + 0.5 * pred_h)

        rect = patches.Rectangle((pred_x1, pred_y1), pred_x2 - pred_x1, pred_y2 - pred_y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.title(f"Green = GT | Red = Predicted | File: {filename}")
plt.axis('off')
plt.tight_layout()
plt.show()

