import numpy as np
import os
from pathlib import Path
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Use cwd if __file__ not defined
try:
    current_dir = Path(__file__).parent
except NameError:
    current_dir = Path.cwd()

root_dir = current_dir.parent / "data"

# Load Data
train_image = np.load(os.path.join(root_dir, 'train_image.npy'))

# Ensure 3-channels
train_image = np.array([cv2.resize(img, (224, 224)) if img.shape[-1]==3 
                        else cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_GRAY2RGB)
                        for img in train_image])

# Preprocess for VGG16
train_img_preprocess = preprocess_input(train_image.astype('float32'))
print(train_img_preprocess.shape)

# Load Model
model_path = os.path.join(current_dir, 'vgg16_model.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = load_model(model_path)

# Get FC2 features
fc2_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('fc2').output)

current_dir = Path(__file__).parent

fc2_feature = fc2_model.predict(train_img_preprocess, batch_size=64, verbose=1)
print(fc2_feature.shape)

fc2_model.save(os.path.join(current_dir,'fc2_model.h5'))
np.save(os.path.join(current_dir, 'fc2_features.npy'), fc2_feature)

