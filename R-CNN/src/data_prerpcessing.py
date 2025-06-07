import numpy as np
import pandas as pd
import os
import cv2
import json
import warnings
from pathlib import Path
from src.utils import get_iou,get_target_bbox
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")

ROOT_DIR= os.getcwd()
image_folder = os.path.join(ROOT_DIR,'PennFudanPed/PNGImages')
annot_folder = os.path.join(ROOT_DIR,'PennFudanPed/boundary_box')

cv2.setUseOptimized(True)
ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation() 



# Sample Image to visualize the 200 region proposal to show case in the image



im = cv2.imread(os.path.join(image_folder,"FudanPed00007.png"))
ss.setBaseImage(im)
ss.switchToSelectiveSearchFast()
rects = ss.process()
imOut = im.copy()
for i, rect in (enumerate(rects)):
    if i >200:
        break
    x, y, w, h = rect
    cv2.rectangle(imOut, (x, y), (x+w, y+h), (255, 255, 0), 1, cv2.LINE_AA)
plt.imshow(imOut)
plt.axis('off')
plt.show()


#++++++++++++++++++++++++++++ Data Preprocess Start Here +++++++++++++++++++++++++++++++++

def process_image(filename):
    local_train_image =[]
    local_train_label = []
    local_target_bbox = []
    local_proposal_bbox=[]
    local_gt_bbox=[]

    try:
        im = cv2.imread(os.path.join(image_folder,filename))  
        if im is None or im.shape[0] == 0 or im.shape[1] == 0:
            print(f"Skipping corrupt or empty image: {filename}")
        annot_file = os.path.join(annot_folder,filename.replace('.png','.json'))
        with open(annot_file,'r') as f:
            annot = json.load(f)
        ground_truth_bbox = annot['boundary_box']
        ss.setBaseImage(im)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()
        imOut = im.copy()
        tp_count,fp_count,tp_flag,fp_flag,flag=0,0,0,0,0
        for i, rect in (enumerate(rects)):
            if i >2000 or flag ==1:
                break
            x, y, w, h = rect
            for annot in ground_truth_bbox:
                bb1= {'x1': annot[0], 'x2': annot[2], 'y1': annot[1], 'y2': annot[3]}
                bb2= {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h}
                iou = get_iou(bb1,bb2)
                try:
                    timage = imOut[y:y + h, x:x + w]

                    if timage is None or timage.shape[0] == 0 or timage.shape[1] == 0:
                        continue

                    resize = cv2.resize(timage, (227, 227), interpolation=cv2.INTER_AREA)
                except:
                    continue
                
                if iou > 0.5 and tp_flag == 0:
                    local_train_image.append(resize)
                    local_train_label.append(1)
                    local_proposal_bbox.append([x, y, x + w, y + h])
                    local_gt_bbox.append(annot)
                    target = get_target_bbox(annot, [x, y, x + w, y + h])
                    local_target_bbox.append(target)
                    tp_count += 1
                    if tp_count > 35:
                        tp_flag = 1
            
                if iou < 0.3 and fp_flag == 0:
                    local_train_image.append(resize)
                    local_train_label.append(0)
                    local_proposal_bbox.append([0,0,0,0])
                    local_gt_bbox.append([0,0,0,0])
                    
                    local_target_bbox.append([0,0,0,0])
                    fp_count += 1
                    if fp_count > 15:
                        fp_flag = 1
                if tp_flag == 1 and fp_flag == 1:
                    flag = 1
    except Exception as e:
        print(f"Error in {filename}: {e}")
    return  local_train_image,local_train_label,local_proposal_bbox,local_gt_bbox,local_target_bbox

'''
train_image = []
train_label = []
proposal_bbox = []
gt_bbox = []
target_bbox = []
for filename in tqdm(os.listdir(image_folder)):
    local_train_image,local_train_label,local_proposal_bbox,local_gt_bbox,local_target_bbox= process_image(filename)
    train_image.extend(local_train_image)
    train_label.extend(local_train_label)
    proposal_bbox.extend(local_proposal_bbox)
    gt_bbox.extend(local_gt_bbox)
    target_bbox.extend(local_target_bbox)

train_image= np.array(train_image)
train_label= np.array(train_label)
proposal_box=np.array(proposal_bbox)
target_box=np.array(target_bbox)
gt_box=np.array(gt_bbox)

out_dir = os.path.join(ROOT_DIR,'R-CNN/data')

np.save(os.path.join(out_dir,'train_image.npy'),train_image)
np.save(os.path.join(out_dir,'train_label.npy'),train_label)
np.save(os.path.join(out_dir,'proposal_box.npy'),proposal_box)
np.save(os.path.join(out_dir,'target_box.npy'),target_box)
np.save(os.path.join(out_dir,'gt_box.npy'),gt_box)

print(len(train_image))
print(np.unique(train_label,return_counts=True))

'''