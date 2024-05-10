#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:15:52 2022

@author: abdul
"""

import tensorflow
import io
from io import BytesIO
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from keras.models import Sequential
from store_image_s3 import create_bucket
from store_image_s3 import extract_image
import shutil
import subprocess
import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker_deploy import create_image_uri
from sagemaker_deploy import define_endpoint

sess = sagemaker.Session()
region = sess.boto_region_name
bucket = sess.default_bucket()

image_uri = create_image_uri()

client = boto3.client(service_name='sagemaker')
runtime = boto3.client(service_name='sagemaker-runtime')

bucket_name = 'resnetbucketsagemaker-amits'
folder = 'sagemaker/content_recommendation'


create_bucket(bucket_name)


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


def load_save_resnet50_model(model_path):
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    resnet50.trainable = False
    shutil.rmtree(model_path, ignore_errors=True)
    model = Sequential()
    model.add(resnet50)
    model.add(GlobalMaxPooling2D())
    model.save(model_path + '.h5', include_optimizer=False)

saved_model_dir = 'resnet50_saved_model' 
model_ver = '1'
model_path = os.path.join(saved_model_dir, model_ver)

load_save_resnet50_model(model_path)

shutil.rmtree('model.tar.gz', ignore_errors=True)
# Define the command as a string
command = "tar cvfz model.tar.gz code -C resnet50_saved_model"

# Run the command
try:
    subprocess.run(command, check=True, shell=True)
    print("Archive created successfully.")
except subprocess.CalledProcessError:
    print("Error occurred while creating the archive.")

prefix = 'keras_models_serverless'
s3_model_path = sess.upload_data(path='model.tar.gz', key_prefix=prefix)


from time import gmtime, strftime

model_name = 'keras-serverless' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print('Model name: ' + model_name)

create_model_response = client.create_model(
    ModelName = model_name,
    Containers=[{
        "Image": image_uri,
        "Mode": "SingleModel",
        "ModelDataUrl": s3_model_path,
    }],
    ExecutionRoleArn ="arn:aws:iam::654654196449:role/service-role/AmazonSageMaker-ExecutionRole-20240504T172417"
)

print("Model Arn: " + create_model_response['ModelArn'])



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
    byteImgIO = io.BytesIO()
    byteImg = Image.open(io.BytesIO(img_path))
    #byteImg = Image.open(img_path)
    byteImg.save(byteImgIO, "PNG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()
    dataBytesIO = io.BytesIO(byteImg)
    Image.open(dataBytesIO)
    img = image.load_img(dataBytesIO,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# # Get ALL file names from Image Folder
# print(os.listdir('img'))

# filenames = []
# for file in os.listdir('img'):
#     filenames.append(os.path.join('img', file))

# print('This is', file)
images = extract_image(bucket_name, folder)
# Append All one dimension (2048X1) to feature list of all IMAGES
feature_list = []
for image_file in tqdm(images):
    feature_list.append(extract_features(image_file, model))

define_endpoint(model_name)

pickle.dump(feature_list, open('embeddings.pkl','wb'))
pickle.dump(images, open('images.pkl','wb'))