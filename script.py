
import tensorflow as tf
import io
import sys
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
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO 
import argparse
import joblib
import os
import numpy as np
import pandas as pd
    
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib")) 
    return clf
    
if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--weights", type=str, default='imagenet')
    parser.add_argument("--include_top", type=bool, default=False)
    parser.add_argument("--input_shape", type=tuple, default=(224,224,3))


    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
   


    args, _ = parser.parse_known_args()
    print("Python Version: ", sys.version)
    print("Tensorflow Version: ", tensorflow.__version__)  # Prints TensorFlow version
    print("Numpy Version: ", np.__version__)
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)
    
    print("Training Resnet Model.....")
    print()
    resnet50 = ResNet50(weights=args.weights, include_top=args.include_top, input_shape=args.input_shape)

    # as model is already trained on Image Net we dont want to train it
    resnet50.trainable = False

    model = Sequential()

    model.add(resnet50)

    # after removing top layer of RESNET 50 we use our custom top layer Global Maxpool2D
    model.add(GlobalMaxPooling2D()) 

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model,model_path)
    print("Model persisted at " + model_path)
    print()


    