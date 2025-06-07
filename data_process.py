import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from PIL import Image
import urllib.request
import zipfile
import os
from pathlib import Path
import json

current_dir = Path(__file__).parent
annotation = os.path.join(current_dir,"boundary_box")
os.makedirs(annotation,exist_ok=True)
image_dir = os.path.join(current_dir, "PNGImages")
mask_dir = os.path.join(current_dir, "PedMasks")


# Define the URL and download path
url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
zip_path = "PennFudanPed.zip"
data_dir = "data"

# Download the zip file
if not os.path.exists(zip_path):
    print("Downloading Penn-Fudan Pedestrian Dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")

# Extract the zip file
if not os.path.exists(data_dir):
    print("Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("Extraction complete.")


# get the boundary box from the mask dir and append in the annotation locaion 
def get_bounding_boxes(mask):
    mask = np.array(mask)
    obj_ids = np.unique(mask)[1:]  # skip background (0)

    boxes = []
    for obj_id in obj_ids:
        pos = np.where(mask == obj_id)
        xmin = int(np.min(pos[1]))
        xmax = int(np.max(pos[1]))
        ymin = int(np.min(pos[0]))
        ymax = int(np.max(pos[0]))
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

for filename in os.listdir(image_dir):
    if not filename.endswith('.png'):
        continue
    img_path = os.path.join(image_dir,filename)
    mask_path =  os.path.join(mask_dir, filename.replace(".png", "_mask.png"))
    mask=Image.open(mask_path)
    boxes=get_bounding_boxes(mask)
    annot = {
        'filename':filename,
        'boundary_box': boxes,
        "class" : ['Pedestrain']*len(boxes)
    }
    annotation_path = os.path.join(annotation,filename.replace('.png','.json'))
    with open(annotation_path,"w") as f:
        json.dump(annot,f)

# visualize a sample image with the bounding box for testing

sample_image = 'FudanPed00007.png'
img_path  = os.path.join(current_dir,'PNGImages',sample_image)
annot_path = os.path.join(annotation,sample_image.replace('.png','.json'))
image=Image.open(img_path)
with open(annot_path,'r') as f:
    annot = json.load(f)
fig,ax = plt.subplots(1)
ax.imshow(image)
for box in annot['boundary_box']:
    x1,y1,x2,y2=box
    rect = Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor='red',facecolor='none')
    ax.add_patch(rect)


plt.title(f"Image: {sample_image} with {len(annot['boundary_box'])} boxes")
plt.axis('off') 
plt.show()
