# ----  Inference ----

import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from src.utils import NMS,apply_deltas


def inference(image, clf, reg, fc2_model):
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pre = preprocess_input(img_rgb.astype('float32'))
    feature = fc2_model.predict(np.expand_dims(img_pre, axis=0))

    is_object = clf.predict(feature)[0]
    prob = clf.predict_proba(feature)[0][1]

    if is_object:
        bbox = reg.predict(feature)[0]
        return bbox, prob
    return None, prob

def inference_with_nms(image, region_proposals, clf, reg, fc2_model, iou_thresh=0.3):
    pred_boxes = []
    scores = []

    for region in region_proposals:
        x1, y1, x2, y2 = region
        roi = image[y1:y2, x1:x2]
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            continue
        roi_resized = cv2.resize(roi, (224, 224))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_pre = preprocess_input(roi_rgb.astype('float32'))
        feature = fc2_model.predict(np.expand_dims(roi_pre, axis=0), verbose=0)

        is_object = clf.predict(feature)[0]
        if is_object:
            score = clf.predict_proba(feature)[0][1]
            bbox = reg.predict(feature)[0]
            pred_box = apply_deltas(region,bbox)
            
            pred_boxes.append(pred_box)
            scores.append(score)

    if pred_boxes:
        idxs = NMS(np.array(pred_boxes), np.array(scores), overlap_thresh=iou_thresh)
        final_boxes = [pred_boxes[i] for i in idxs]
        final_scores = [scores[i] for i in idxs] 
        print('prediction_box:',final_boxes)
        return final_boxes, final_scores
    return [], []

