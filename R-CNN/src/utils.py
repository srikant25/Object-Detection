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

def get_target_bbox(gt,prop):
    x1,y1,x2,y2=prop[0],prop[1],prop[2],prop[3]
    tx = (gt[0] - x1) / (x2 - x1)
    ty = (gt[1] - y1) / (y2 - y1)
    tw = np.log((gt[2] - gt[0]) / (x2 - x1))
    th = np.log((gt[3] - gt[1]) / (y2 - y1))
    bbox = [tx, ty, tw, th]
    return bbox

def apply_deltas(prop, deltas):
    x1, y1, x2, y2 = prop
    prop_w = x2 - x1
    prop_h = y2 - y1

    tx, ty, tw, th = deltas
    pred_x1 = x1 + tx * prop_w
    pred_y1 = y1 + ty * prop_h
    pred_x2 = pred_x1 + np.exp(tw) * prop_w
    pred_y2 = pred_y1 + np.exp(th) * prop_h

    return [int(pred_x1), int(pred_y1), int(pred_x2), int(pred_y2)]


def NMS(boxes, scores, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = boxes.astype("float")
    pick = []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area[last] + area[idxs[:-1]] - inter)

        idxs = idxs[np.where(ovr <= overlap_thresh)]

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