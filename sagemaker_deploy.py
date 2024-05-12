# import shutil
# import subprocess
# import sagemaker
# import boto3
# from sagemaker import image_uris
# from time import gmtime, strftime


# client = boto3.client(service_name='sagemaker')
# runtime = boto3.client(service_name='sagemaker-runtime')
# sess = sagemaker.Session()
# #role = sagemaker.get_execution_role()
# region = sess.boto_region_name
# bucket = sess.default_bucket()





# #Define endpoint configuration
# import time

# def define_endpoint(s3_model_path):
#     #Container Image url
#     framework_name = "tensorflow"
#     framework_version = "1.15.4"
#     image_uri = sagemaker.image_uris.retrieve(
#     framework=framework_name,
#     region=region,
#     version=framework_version,
#     py_version="py3",
#     image_scope='inference',
#     instance_type='ml.c5.xlarge')

#     model_name = 'keras-serverless' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
#     print('Model name: ' + model_name)  

#     create_model_response = client.create_model(
#         ModelName = model_name,
#         Containers=[{
#             "Image": image_uri,
#             "Mode": "SingleModel",
#             "ModelDataUrl": s3_model_path,
#         }],
#         ExecutionRoleArn ="arn:aws:iam::654654196449:role/service-role/AmazonSageMaker-ExecutionRole-20240504T172417"
#     )

#     print("Model Arn: " + create_model_response['ModelArn'])

#     #Define endpoint configuration
#     keras_epc_name = "keras-serverless-epc" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

#     endpoint_config_response = client.create_endpoint_config(
#     EndpointConfigName=keras_epc_name,
#     ProductionVariants=[
#         {
#             "VariantName": "kerasVariant",
#             "ModelName": model_name,
#             "ServerlessConfig": {
#                 "MemorySizeInMB": 4096,
#                 "MaxConcurrency": 1,
#             }
#         },
#     ],
#     )

#     print("Serverless Endpoint Configuration Arn: " + endpoint_config_response['EndpointConfigArn'])

#     endpoint_name = "keras-serverless-ep" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

#     #Create an endpoint

#     create_endpoint_response = client.create_endpoint(
#         EndpointName=endpoint_name,
#         EndpointConfigName=keras_epc_name,
#     )

#     print('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])

#     # wait for endpoint to reach a terminal state (InService) using describe endpoint
#     describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)

#     while describe_endpoint_response["EndpointStatus"] == "Creating":
#         describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
#         print(describe_endpoint_response["EndpointStatus"])
#         time.sleep(15)