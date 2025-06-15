import os
import cv2
import json
import pickle
import numpy as np
from tqdm import tqdm
from utils import get_iou, get_target_bbox

root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'PennFudanPed/PNGImages')
bbox_folder = os.path.join(root_dir, 'PennFudanPed/boundary_box')
resized_image_folder = os.path.join(root_dir, 'Fast-RCNN/resized_images')
output_dir = os.path.join(root_dir, 'Fast-RCNN/data')

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

os.makedirs(resized_image_folder, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def resize_image(image, short_side=600):
    h, w = image.shape[:2]
    scale = short_side / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img, scale

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
        resized_im, scale = resize_image(im, short_side=600)

        # Save resized image to disk
        resized_path = os.path.join(resized_image_folder, filename)
        cv2.imwrite(resized_path, resized_im)

        # Scale ground truth boxes
        scaled_gt_boxes = [[x1 * scale, y1 * scale, x2 * scale, y2 * scale] for (x1, y1, x2, y2) in ground_truth_bboxes]

        ss.setBaseImage(resized_im)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        tp_count = 0
        fp_count = 0

        for i, rect in enumerate(rects):
            if i > 3000 or (tp_count >= 30 and fp_count >= 30):
                break

            x, y, rw, rh = rect
            bb2 = {'x1': x, 'x2': x + rw, 'y1': y, 'y2': y + rh}
            max_iou = 0
            best_gt = None

            for gt in scaled_gt_boxes:
                bb1 = {'x1': gt[0], 'x2': gt[2], 'y1': gt[1], 'y2': gt[3]}
                iou = get_iou(bb1, bb2)
                if iou > max_iou:
                    max_iou = iou
                    best_gt = bb1

            box = {
                'file_name': filename,
                'region_proposal_box': [x, y, x + rw, y + rh],
                'ground_truth_box': [0, 0, 0, 0],
                'target_box': [0, 0, 0, 0],
                'label': 0
            }

            if max_iou >= 0.5 and tp_count < 30:
                box['ground_truth_box'] = [best_gt['x1'], best_gt['y1'], best_gt['x2'], best_gt['y2']]
                box['target_box'] = get_target_bbox(box['ground_truth_box'], box['region_proposal_box'])
                box['label'] = 1
                tp_count += 1
                region_proposals.append(box)

            elif max_iou < 0.3 and fp_count < 30:
                fp_count += 1
                region_proposals.append(box)

    except Exception as e:
        print(f"Error in {filename}: {e}")

    return region_proposals

def data_process():
    all_proposals = []

    for filename in tqdm(os.listdir(image_folder)):
        if filename.endswith('.png'):
            proposals = process_image(filename)
            all_proposals.extend(proposals)

    # Save region proposals
    with open(os.path.join(output_dir, 'region_proposals.pkl'), 'wb') as f:
        pickle.dump(all_proposals, f)

    print(f"Total region proposals: {len(all_proposals)}")
    print(f"Saved region proposals to: {os.path.join(output_dir, 'region_proposals.pkl')}")
    print(f"Resized images saved to: {resized_image_folder}")

# if __name__ == "__main__":
#     data_process()
for image in os.listdir(image_folder):
    im = np.array(cv2.imread(os.path.join(image_folder,image)))
    print(im.shape[:-1])

