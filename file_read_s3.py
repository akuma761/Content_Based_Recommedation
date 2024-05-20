import boto3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import io  # Import io for BytesIO functionality

# Load environment variables
load_dotenv()

# Create an S3 client
s3_client = boto3.client('s3')
bucket_name = 'your-bucket-name'  # replace with your bucket name
folder_name = 'your-folder-name/'  # replace with your folder path in the bucket

def fetch_filenames(bucket, prefix):
    """Fetch all filenames in the S3 bucket under a specific prefix."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' in response:
            files = [item['Key'] for item in response['Contents'] if item['Key'] != prefix]
            return files
        return []
    except NoCredentialsError:
        print("Credentials not available")
        return []

filenames = fetch_filenames(bucket_name, folder_name)


def download_and_display_images(filenames, indices):
    for index in indices:
        file_key = filenames[index]
        try:
            # Create a bytes buffer
            img_buffer = io.BytesIO()

            # Download the image file into the buffer
            s3_client.download_fileobj(bucket_name, file_key, img_buffer)
            
            # Move the buffer's cursor to the start
            img_buffer.seek(0)
            
            # Convert buffer to NumPy array for use with OpenCV
            file_bytes = np.asarray(bytearray(img_buffer.read()), dtype=np.uint8)
            
            # Decode the image
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Display the image
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
            
        except Exception as e:
            print(f"Failed to download or display {file_key}: {str(e)}")

# Example use
filenames = fetch_filenames(bucket_name, folder_name)  # Make sure you have this function defined as shown previously
indices = [0, 1, 2]  # Example indices
download_and_display_images(filenames, indices)
