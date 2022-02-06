#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 16:01:08 2022

@author: abdul
"""

import pickle
import numpy as np
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
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt 


feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

filenames = pickle.load(open('filenames.pkl', 'rb'))

print(feature_list.shape)

#==============================================================================================

resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

# as model is already trained on Image Net we dont want to train it
resnet50.trainable = False

model = Sequential()

model.add(resnet50)

# after removing top layer of RESNET 50 we use our custom top layer Global Maxpool2D
model.add(GlobalMaxPooling2D())


#==============================================================================================
# EXRACTING 1-D FEATURES OF A TEST IMAGE
img = image.load_img('./test/19933.jpg', target_size = (224, 224))
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


#=================NEAREST NEIGHBOUR ALGORITHM================================================

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([norm_pred])

indices = indices[0][1:6]

print(indices)


for i in range (0, len(indices)):
    temp_img = cv2.imread(filenames[indices[i]])
    plt.imshow(cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    