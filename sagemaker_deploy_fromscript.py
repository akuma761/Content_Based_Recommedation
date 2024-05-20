from sagemaker.sklearn.estimator import SKLearn
from time import gmtime, strftime
import boto3
import tqdm
import pickle
from tqdm import tqdm
import io
import numpy as np
from io import BytesIO
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input



import sagemaker
from sagemaker.tensorflow import TensorFlow
from preprocessing import extract_features
from store_image_s3 import create_bucket
from store_image_s3 import extract_image

sess = sagemaker.Session()
region = sess.boto_region_name
bucket = sess.default_bucket()

#image_uri = create_image_uri()

client = boto3.client(service_name='sagemaker')
runtime = boto3.client(service_name='sagemaker-runtime')

bucket_name = 'resnetbucketsagemaker-amits'
folder = 'sagemaker/content_recommendation'

create_bucket(bucket_name)



# Role and other configurations
role = "arn:aws:iam::654654196449:role/service-role/AmazonSageMaker-ExecutionRole-20240504T172417"
FRAMEWORK_VERSION = '2.3.0'  # Specify your TensorFlow version

# Create TensorFlow estimator
tf_estimator = TensorFlow(
    entry_point='script.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='2.3.0',
    py_version='py37',
    script_mode=True,
    base_job_name='resnet50-tensorflow',
    hyperparameters={'weights': 'imagenet', 'include_top': False,'epochs': 10 },
    use_spot_instances=True,
    max_wait=7200,
    max_run=3600
)

tf_estimator.hyperparameters()

# Fit model

# print('This is', file)
images = extract_image(bucket_name, folder)
# Append All one dimension (2048X1) to feature list of all IMAGES
def process_image(img_path):
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
    return preprocessed_img

def fit_model(img_paths, model):
    """Fit the model using a list of image paths and corresponding labels."""
    images = np.vstack([process_image(img_paths)])
    labels = np.array([1])
    model.fit(images, labels)

images = extract_image(bucket_name, folder)
 

for image_file in tqdm(images):
    fit_model(image_file, tf_estimator)

# pickle.dump(feature_list, open('embeddings.pkl','wb'))
# pickle.dump(images, open('images.pkl','wb'))
# launch training job, with asynchronous call
# tf_estimator.fit({"train": trainpath, "test": testpath}, wait=True)
# sklearn_estimator.fit({"train": datapath}, wait=True)


tf_estimator.latest_training_job.wait(logs="None")
artifact = client.describe_training_job(
    TrainingJobName=tf_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact persisted at " + artifact)

print(artifact)

# from sagemaker.sklearn.model import SKLearnModel
# from time import gmtime, strftime
from sagemaker.tensorflow import TensorFlowModel

model_name = "Custom-tensorflow-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
model = TensorFlowModel(
    name =  model_name,
    model_data=artifact,
    role="arn:aws:iam::566373416292:role/service-role/AmazonSageMaker-ExecutionRole-20230120T164209",
    entry_point="script.py",
    framework_version=FRAMEWORK_VERSION,
)

##Endpoints deployment
endpoint_name = "Custom-tensorflow-model-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("EndpointName={}".format(endpoint_name))

predictor = tf_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m4.xlarge",
    endpoint_name=endpoint_name,
)

endpoint_name
