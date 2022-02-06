#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:15:52 2022

@author: abdul
"""

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from keras.models import Sequential


# we will use Image Net weights and Add our custom top Layer
# (224, 224, 3) is the standard size while doing Transfer Learning
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# as model is already trained on Image Net we dont want to train it
resnet50.trainable = False

model = Sequential()

model.add(resnet50)

# after removing top layer of RESNET 50 we use our custom top layer Global Maxpool2D
model.add(GlobalMaxPooling2D())

# after prinitng summary we see that there are ZERO Trainable Parameters
print(model.summary())

'''
# EXRACTING 1-D FEATURES OF A SINGLE IMAGE

img = image.load_img('./images/1165.jpg', target_size = (224, 224))
img_array = image.img_to_array(img)
img_array.shape
print(img_array)

# RESIZING image size ADDING BATCH No.
expanded_img = np.expand_dims(img_array, axis = 0)
expanded_img.shape

# processing the given image to sychronize standard as ImageNet dataset
# images are converted from RGB to BGR and each color channel is ZERO-CENTERED wrt to ImageNet Dataset without scaling
processed_img = preprocess_input(expanded_img)

# now using RESNET to get prediction on a given Image
pred = model.predict(processed_img)
pred.shape

pred = pred.flatten()
pred.shape

norm_pred = pred/norm(pred)

'''

def extract_features(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Get ALL file names from Image Folder
print(os.listdir('images'))

filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))

# Append All one dimension (2048X1) to feature list of all IMAGES
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl','wb'))
pickle.dump(filenames, open('filenames.pkl','wb'))


