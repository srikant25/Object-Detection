import numpy as np
import pandas as pd
import os 
from PIL import Image
import cv2
import warnings
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import mixed_precision

current_dir = Path(__file__).parent
root_dir = current_dir.parent/"data"

train_image = np.load(os.path.join(root_dir,'train_image.npy'))
train_label =np.load(os.path.join(root_dir,'train_label.npy'))
target_box= np.load(os.path.join(root_dir,'target_box.npy'))
gt_box=np.load(os.path.join(root_dir,'gt_box.npy'))
proposal_box = np.load(os.path.join(root_dir,'proposal_box.npy'))
train_image = np.array([cv2.resize(img, (224, 224)) for img in train_image])


mixed_precision.set_global_policy('mixed_float16')

vgg16 = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
for layer in vgg16.layers[:-2]:
    layer.trainable=False
x= vgg16.get_layer('fc2')
last_layer = x.output
x= tf.keras.layers.Dense(1,activation='sigmoid')(last_layer)
'''
model = tf.keras.Model(vgg16.input,x)
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['acc'])

# model.summary()

model.fit(train_image, train_label,epochs=5 , batch_size=4,verbose=1,validation_split=0.2,shuffle=True)
current_dir = Path(__file__).parent
model.save(os.path.join(current_dir,'vgg16_model.h5'))
'''


model = load_model(os.path.join(current_dir,'vgg16_model.h5'))

fc2_model = tf.keras.Model(inputs=model.input,outputs = model.get_layer('fc2').output)

train_img_preprocess= preprocess_input(train_image.astype('float32'))
fc2_feature=fc2_model.predict(train_img_preprocess,batch_size=64,verbose=1)
print(fc2_feature.shape)