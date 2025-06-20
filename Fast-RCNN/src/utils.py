import numpy as np
from sklearn.metrics import average_precision_score

def get_iou(bb1,bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1 ['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2 ['y2']
    x_left = max(bb1['x1'],bb2['x1'])
    x_right = min(bb1['x2'],bb2['x2'])
    y_top = min(bb1['y2'],bb2['y2'])
    y_bottom = max(bb1['y1'],bb2['y1'])
    if x_right < x_left or y_top<y_bottom :
        return 0.0
    intersection_area = (x_right-x_left)*(y_top-y_bottom)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area/float(bb1_area+bb2_area-intersection_area)
    assert iou >=0.0
    assert iou <=1.0
    return iou

import numpy as np

def roi_pooling(feature_map, roi, output_size=(7, 7), spatial_scale=14/512):
    """
    Simulated RoI Pooling operation from scratch using numpy.

    Args:
        feature_map: np.ndarray of shape (H, W, C), e.g., (14, 14, 512)
        roi: list or np.array of [x1, y1, x2, y2] on original image scale (e.g., 512x512)
        output_size: tuple (h, w) of output pooled feature size
        spatial_scale: scale from input image to feature map (e.g., 1/16 for VGG16)

    Returns:
        pooled: np.ndarray of shape (output_h, output_w, C)
    """
    x1, y1, x2, y2 = [int(coord * spatial_scale) for coord in roi]

    # Clip to feature map bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(feature_map.shape[1]-1, x2), min(feature_map.shape[0]-1, y2)

    roi_width = max(x2 - x1 + 1, 1)
    roi_height = max(y2 - y1 + 1, 1)

    pooled_height, pooled_width = output_size
    bin_size_w = roi_width / pooled_width
    bin_size_h = roi_height / pooled_height

    channels = feature_map.shape[2]
    pooled = np.zeros((pooled_height, pooled_width, channels), dtype=np.float32)

    for ph in range(pooled_height):
        for pw in range(pooled_width):
            h_start = int(np.floor(y1 + ph * bin_size_h))
            h_end = int(np.ceil(y1 + (ph + 1) * bin_size_h))
            w_start = int(np.floor(x1 + pw * bin_size_w))
            w_end = int(np.ceil(x1 + (pw + 1) * bin_size_w))

            h_start, w_start = max(h_start, 0), max(w_start, 0)
            h_end, w_end = min(h_end, feature_map.shape[0]), min(w_end, feature_map.shape[1])

            if h_end <= h_start or w_end <= w_start:
                continue  # empty region

            region = feature_map[h_start:h_end, w_start:w_end, :]
            pooled[ph, pw, :] = np.max(region, axis=(0, 1))

    return pooled


def get_target_bbox(gt, prop):
    px, py, pw, ph = prop_to_cxcywh(prop)
    gx, gy, gw, gh = prop_to_cxcywh(gt)

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = np.log(gw / pw)
    th = np.log(gh / ph)
    return [tx, ty, tw, th]

def prop_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h




def apply_deltas(prop, deltas):
    assert len(prop) == 4
    assert len(deltas) == 4 
    x1, y1, x2, y2 = prop
    prop_w = x2 - x1
    prop_h = y2 - y1
    prop_ctr_x = x1 + 0.5 * prop_w
    prop_ctr_y = y1 + 0.5 * prop_h

    tx, ty, tw, th = deltas
    pred_ctr_x = prop_ctr_x + tx * prop_w
    pred_ctr_y = prop_ctr_y + ty * prop_h
    pred_w = np.exp(tw) * prop_w
    pred_h = np.exp(th) * prop_h

    pred_x1 = pred_ctr_x - 0.5 * pred_w
    pred_y1 = pred_ctr_y - 0.5 * pred_h
    pred_x2 = pred_ctr_x + 0.5 * pred_w
    pred_y2 = pred_ctr_y + 0.5 * pred_h

    return [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]





def NMS(boxes, scores, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")
    pick = []

    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)[::-1]  # Descending order

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (area[i] + area[idxs[1:]] - inter)

        idxs = idxs[np.where(iou <= overlap_thresh)[0] + 1]

    return pick


def evaluate_mAP(y_true_cls, y_pred_cls, y_true_box, y_pred_box):
    cls_ap = average_precision_score(y_true_cls, y_pred_cls)
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)
    iou_scores = [iou(p, t) for p, t in zip(y_pred_box, y_true_box)]
    mean_iou = np.mean(iou_scores)
    return cls_ap, mean_iou


import cv2
import matplotlib.pyplot as plt

def visualize(image, pred_boxes=None, true_boxes=None, label='object', scores=None):
    image = image.copy()
     
    # Draw ground truth boxes (can be one or many)
    if true_boxes is not None:
        true_boxes = np.array(true_boxes)
        if len(true_boxes.shape) == 2:  # multiple GT boxes
            for box in true_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        elif len(true_boxes.shape) == 1:  # single GT box
            x1, y1, x2, y2 = map(int, true_boxes)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw predicted boxes (can be one or many)
    if pred_boxes is not None:
        if isinstance(pred_boxes[0], (list, tuple, np.ndarray)):
            for i, box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, box)
                score = scores[i] if scores is not None and i < len(scores) else None
                text = f'{label}: {score:.2f}' if score is not None else label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green for prediction
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            x1, y1, x2, y2 = map(int, pred_boxes)
            score = scores[0] if scores else None
            text = f'{label}: {score:.2f}' if score is not None else label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

