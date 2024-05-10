import sagemaker
import boto3
from PIL import Image
import io

bucket_name = 'resnetbucketsagemaker-amits'
folder = 'sagemaker/content_recommendation'

def create_bucket(bucket_name):
    # Create an S3 bucket if it does not already exist
    session = boto3.Session()
    s3client = session.client('s3')

    # Check if the bucket already exists
    try:
        s3client.head_bucket(Bucket=bucket_name)
        print(f"Bucket {bucket_name} already exists.")
    except s3client.exceptions.ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:  # Bucket does not exist
            s3client.create_bucket(Bucket=bucket_name,
                                   CreateBucketConfiguration={
                                       'LocationConstraint': session.region_name})
            print(f"Created bucket {bucket_name} in region {session.region_name}.")
        else:
            raise

    # Initialize SageMaker client and session
    sm_boto3 = boto3.client("sagemaker")
    sess = sagemaker.Session()

    # Upload data to S3. SageMaker will take image data from S3
    sk_prefix = "sagemaker/content_recommendation"
    try:
        files = sess.upload_data(
            path="img",  # Ensure this path exists and contains data to upload
            bucket=bucket_name,
            key_prefix=sk_prefix
        )
        print(f"Files uploaded to {bucket_name}/{sk_prefix}")
    except Exception as e:
        print(f"Failed to upload data: {str(e)}")

# def create_bucket(bucket_name):
#     # Create an S3 bucket
#     session = boto3.Session(region_name='us-east-1')

#     #bucket_name = 'resnetbucketsagemaker-amits' # Mention the created S3 bucket name here
#     print("Using bucket " + bucket_name)

#     s3client = session.client('s3')
#     s3client.create_bucket(Bucket=bucket_name)

#     sm_boto3 = boto3.client("sagemaker")
#     sess = sagemaker.Session()
#     region = sess.boto_session.region_name    

#     # send data to S3. SageMaker will take image data from s3
#     sk_prefix = "sagemaker/content_recommendation"
#     files = sess.upload_data(
#     path="img", bucket=bucket_name, key_prefix=sk_prefix
#     )

    





def extract_image(bucket_name, folder):
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    bucket = resource.Bucket(bucket_name)
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder}/", StartAfter=f"{folder}/")
    images = []
    # Loop through the objects in the folder
    for obj in response.get('Contents', []):
        key = obj['Key']
        print(key)
        if key.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
            image_obj = bucket.Object(key).get()
            file_stream = image_obj['Body'].read()
            images.append(file_stream)
            #print(images)  # Store each image file stream in a list


    return images

a = extract_image(bucket_name, folder)
#print(a)

if a:
    first_image_data = a[0]  # Access the first element
    image = Image.open(io.BytesIO(first_image_data))
    image.show()  # This will display the image if you are in an environment that supports it
else:
    print("No data available.")

       # Safe access to 'Contents' in case it's missing
     #print(obj['Key'])
    
