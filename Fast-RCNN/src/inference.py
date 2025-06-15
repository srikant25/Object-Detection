import os
import cv2
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as T
from torchvision.ops import nms
from collections import defaultdict
from model import FastRCNN

# -------- CONFIG --------
root_dir = os.getcwd()
PROPOSAL_PATH = os.path.join(root_dir,'Fast-RCNN/data/region_proposals.pkl')
IMAGE_FOLDER = os.path.join(root_dir,'Fast-RCNN/resized_images')
MODEL_PATH = os.path.join("saved_models", "fast_rcnn_best.pth")
IMAGE_IDX = 25                          # Index of image to test
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

# Group proposals by image name
grouped = defaultdict(list)
for p in proposals:
    grouped[p['file_name']].append(p)

# Pick one image
filename = list(grouped.keys())[IMAGE_IDX]
image_path = os.path.join(IMAGE_FOLDER, filename)
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess image
transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

# Gather RoIs and ground truth boxes
samples = grouped[filename]
rois = []
gt_boxes = []

for sample in samples:
    x1, y1, x2, y2 = sample['region_proposal_box']
    rois.append([0, x1, y1, x2, y2])  # batch index = 0
    gt_boxes.append(sample['ground_truth_box'])

rois = torch.tensor(rois, dtype=torch.float32).to(DEVICE)

# -------- Inference --------
with torch.no_grad():
    probs, bbox_deltas = model.predict(image_tensor, rois)

probs = probs.cpu().numpy()
bbox_deltas = bbox_deltas.cpu().numpy()
rois_np = rois.cpu().numpy()

# -------- Decode bboxes --------
pred_boxes = []
scores = []

for i in range(len(rois_np)):
    class_id = np.argmax(probs[i])
    confidence = probs[i][class_id]

    if class_id == 1 and confidence > CONFIDENCE_THRESHOLD:
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

        pred_x1 = pred_ctr_x - 0.5 * pred_w
        pred_y1 = pred_ctr_y - 0.5 * pred_h
        pred_x2 = pred_ctr_x + 0.5 * pred_w
        pred_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes.append([pred_x1, pred_y1, pred_x2, pred_y2])
        scores.append(confidence)

# Convert to tensors for NMS
pred_boxes_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
scores_tensor = torch.tensor(scores, dtype=torch.float32)

# Apply NMS
keep_indices = nms(pred_boxes_tensor, scores_tensor, iou_threshold=IOU_THRESHOLD)

# -------- Visualization --------
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image_rgb)

# Draw ground truth boxes (green)
for box in gt_boxes:
    x1, y1, x2, y2 = map(int, box)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

# Draw predicted boxes after NMS (red)
for idx in keep_indices:
    x1, y1, x2, y2 = map(int, pred_boxes[idx])
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.title(f"GT (Green) | Predicted (Red) | File: {filename}")
plt.axis('off')
plt.tight_layout()
plt.show()

# Print summary
print(f"Total predictions before NMS: {len(pred_boxes)}")
print(f"Predictions after NMS: {len(keep_indices)}")
