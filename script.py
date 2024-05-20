import tensorflow as tf
import argparse
import numpy as np
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications import ResNet50
import joblib

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib")) 
    return clf
    
if __name__ == "__main__":

   if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--weights", type=str, default='imagenet')
    parser.add_argument("--include_top", type=bool, default=False)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--channels", type=int, default=3)
    #parser.add_argument('--epochs', type=int, default=10)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    print("Python Version:", sys.version)
    print("Tensorflow Version:", tf.__version__)
    print("Numpy Version:", np.__version__)

    print("Training Resnet Model...")
    input_shape = (args.width, args.height, args.channels)
    resnet50 = ResNet50(weights=args.weights, include_top=args.include_top, input_shape=input_shape)
    resnet50.trainable = False

    model = Sequential([resnet50, GlobalMaxPooling2D()])

    model_path = os.path.join(args.model_dir, "model")
    model.save(model_path)
    print("Model persisted at " + model_path)

    