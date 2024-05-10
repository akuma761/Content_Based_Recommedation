import boto3
from PIL import Image
import io

bucket_name = 'resnetbucketsagemaker-amits'
folder = 'sagemaker/content_recommendation'
resource = boto3.resource('s3')
bucket = resource.Bucket(bucket_name)
print(bucket.objects)
client = boto3.client('s3')


# Using list_objects_v2 with StartAfter
response = client.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder}/", StartAfter=f"{folder}/")

# Iterate through the returned objects and print keys
for obj in response.get('Contents', []):  # Safe access to 'Contents' in case it's missing
    print(obj['Key'])

# Create a list of files under the specified folder
#files_in_s3 = [obj.key.split(folder + "/")[1] for obj in bucket.objects.filter(Prefix=f"{folder}/") if '/' in obj.key.split(folder + "/")[-1]]

# Create a list of file names under the specified folder
files_in_s3 = [obj.key.split('/')[-1] for obj in bucket.objects.filter(Prefix=f"{folder}/")]

print(files_in_s3)

obj = bucket.Object('sagemaker/content_recommendation/cnn1.jpeg').get()
# Get the body of the object (the file data)
file_stream = obj['Body'].read()

# Convert the byte stream to an image (using PIL for image handling)
image = Image.open(io.BytesIO(file_stream))

# If you need to display the image (for example, in a Jupyter notebook):
image.show()

        