import shutil
import subprocess
import sagemaker
import boto3
from sagemaker import image_uris
from time import gmtime, strftime


client = boto3.client(service_name='sagemaker')
runtime = boto3.client(service_name='sagemaker-runtime')
sess = sagemaker.Session()
#role = sagemaker.get_execution_role()
region = sess.boto_region_name
bucket = sess.default_bucket()



#Container Image url
framework_name = "tensorflow"
framework_version = "1.15.4"

def create_image_uri():
    image_uri = sagemaker.image_uris.retrieve(
        framework=framework_name,
        region=region,
        version=framework_version,
        py_version="py3",
        image_scope='inference',
        instance_type='ml.c5.xlarge'
)
    return image_uri



#Define endpoint configuration
import time

def define_endpoint(model_name):
    keras_epc_name = "keras-serverless-epc" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName=keras_epc_name,
    ProductionVariants=[
        {
            "VariantName": "kerasVariant",
            "ModelName": model_name,
            "ServerlessConfig": {
                "MemorySizeInMB": 4096,
                "MaxConcurrency": 1,
            }
        },
    ],
    )

    print("Serverless Endpoint Configuration Arn: " + endpoint_config_response['EndpointConfigArn'])

    endpoint_name = "keras-serverless-ep" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    create_endpoint_response = client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=keras_epc_name,
    )

    print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])
    describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)

    while describe_endpoint_response["EndpointStatus"] == "Creating":
        describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
        print(describe_endpoint_response["EndpointStatus"])
        time.sleep(15)