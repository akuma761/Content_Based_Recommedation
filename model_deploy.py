from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
import sagemaker
import boto3

container=sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "0.90-1")
print(container)

model_name = 'mymodel123'
model_location= "s3://sagemaker-us-east-1-654654196449/keras_models_serverless/model.tar.gz"
role = "arn:aws:iam::654654196449:role/service-role/AmazonSageMaker-ExecutionRole-20240504T172417"


env = {
    # 'SAGEMAKER_REQUIREMENTS': 'requirements.txt',
    # 'SAGEMAKER_PROGRAM': 'inference.py',
    'SAGEMAKER_SUBMIT_DIRECTORY' : model_location
    }

model= Model(
    model_data=model_location, 
    image_uri =container,
    env = env
)

endpoint_name = model_name

pipeline_model = PipelineModel(name=model_name,
                               role=role,
                               models=[
                                    model
                               ])

pm = pipeline_model.deploy(initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name)
