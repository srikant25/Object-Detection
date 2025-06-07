import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import NMS
from src.data_prerpcessing import process_image
from src.inference import inference_with_nms
from src.visualize import visualize
import cv2
import joblib
from tensorflow.keras.models import load_model

ROOT_DIR= os.getcwd()
image_folder = os.path.join(ROOT_DIR,'PennFudanPed/PNGImages')
annot_folder = os.path.join(ROOT_DIR,'PennFudanPed/boundary_box')
model_path =os.path.join(ROOT_DIR,'R-CNN/src')
fc2_model_path = os.path.join(model_path,'fc2_model.h5')
clf_path = os.path.join(model_path,'svm_classifier.pkl')
reg_path = os.path.join(model_path,'svr_regressor.pkl')
fc2_model= load_model(fc2_model_path)
clf = joblib.load(clf_path)
reg = joblib.load(reg_path)


filename ="FudanPed00008.png"
image = cv2.imread(os.path.join(image_folder,filename))
train_image,train_label,region_proposal,gt_bbox,target_bbox= process_image(filename)
print((gt_bbox))
final_boxes, final_scores = inference_with_nms(image, region_proposal, clf, reg, fc2_model)
print(final_boxes)
print(final_scores)
#for box, score in zip(final_boxes, final_scores):

visualize(image, final_boxes, gt_bbox, label='object', scores=final_scores)




'''
model_path

sample_idx = 0
sample_img = train_image[sample_idx]  # already resized above
sample_gt_box = target_box[sample_idx]
pred_box, prob = inference(sample_img, clf, reg, fc2_model)
visualize(sample_img, pred_box, true_box=sample_gt_box, label='object', score=prob)


sample_idx = 0
sample_img = train_image[sample_idx]
sample_gt_box = target_box[sample_idx]

# Dummy region proposals (replace with selective search later)
region_proposals = [
    [30, 40, 160, 180],
    [35, 45, 158, 178],
    [100, 100, 200, 200],
]

final_boxes, final_scores = inference_with_nms(sample_img, region_proposals, clf, reg, fc2_model)

for box, score in zip(final_boxes, final_scores):
    visualize(sample_img, box, true_box=sample_gt_box, label='object', score=score)

'''

