import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import json
from utils import get_iou

root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'PennFudanPed/PNGImages')
bbox_folder = os.path.join(root_dir, 'PennFudanPed/boundary_box')

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()




def process_image(filename):
    region_proposals = []
    try:
        image_path = os.path.join(image_folder, filename)
        im = cv2.imread(image_path)
        
        if im is None or im.shape[0] == 0 or im.shape[1] == 0:
            print(f"Skipping corrupt or empty image: {filename}")
            return []

        annot_file = os.path.join(bbox_folder, filename.replace('.png', '.json'))
        with open(annot_file, 'r') as f:
            annot = json.load(f)
        ground_truth_bboxes = annot['boundary_box']

        ss.setBaseImage(im)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        tp_count = 0
        fp_count = 0
        used_regions = set()

        for i, rect in enumerate(rects):
            if i > 2000 or (tp_count >= 16 and fp_count >= 48):
                break

            x, y, w, h = rect
            if w < 20 or h < 20:  # skip tiny boxes
                continue

            bb2 = {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h}

            max_iou = 0
            best_gt = None

            for gt in ground_truth_bboxes:
                bb1 = {'x1': gt[0], 'x2': gt[2], 'y1': gt[1], 'y2': gt[3]}
                iou = get_iou(bb1, bb2)
                if iou > max_iou:
                    max_iou = iou
                    best_gt = bb1

            # Only take distinct proposals
            region_key = (x, y, x + w, y + h)
            if region_key in used_regions:
                continue
            used_regions.add(region_key)

            box = {}
            box['file_name'] = filename
            box['region_proposal_box'] =  [x, y, x + w, y + h]

            if max_iou >= 0.5 and tp_count < 16:
                box['ground_truth_box'] =  [gt[0], gt[1], gt[2], gt[3]] if best_gt else []
                box['label'] = 1
                region_proposals.append(box)
                tp_count += 1

            elif 0.1 < max_iou < 0.5 and fp_count < 48:
                box['ground_truth_box'] = {}
                box['label'] = 0
                region_proposals.append(box)
                fp_count += 1
    except Exception as e:
        print(f"Error in {filename}: {e}")
    return  region_proposals

def data_process():
    all_proposals = []
    for filename in tqdm(os.listdir(image_folder)):
        region_proposal= process_image(filename)
        all_proposals.extend(region_proposal)

    all_proposals= np.array(all_proposals)

    out_dir = os.path.join(root_dir,'Fast-RCNN/data')
    os.makedirs(out_dir,exist_ok=True)

    # np.save(os.path.join(out_dir,'region_proposals.npy'),all_proposals)
    import pickle

    with open(os.path.join(out_dir, 'region_proposals.pkl'), 'wb') as f:
        pickle.dump(all_proposals, f)
    
    print(f"Total region proposals: {len(all_proposals)}")
    print(f" Saved to: {os.path.join(out_dir,'region_proposals.pkl')}")
 

if __name__=="__main__":
    data_process()
    


