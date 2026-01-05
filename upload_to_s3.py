# upload_to_s3.py
"""
Upload dataset files to AWS S3 bucket.
"""

import boto3
import os
from botocore.exceptions import ClientError


def upload_to_s3(
    local_dir: str = "data",
    bucket_name: str = "YOUR-BUCKET-NAME",  # <-- CHANGE THIS
    s3_prefix: str = "data"
) -> None:
    """
    Upload all CSV files from local directory to S3.
    
    Args:
        local_dir: Local directory containing files
        bucket_name: S3 bucket name
        s3_prefix: S3 folder prefix
    """
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Get all CSV files
    files = [f for f in os.listdir(local_dir) if f.endswith('.csv')]
    
    print(f"Uploading {len(files)} files to s3://{bucket_name}/{s3_prefix}/")
    print("-" * 50)
    
    for filename in files:
        local_path = os.path.join(local_dir, filename)
        s3_key = f"{s3_prefix}/{filename}"
        
        try:
            s3_client.upload_file(local_path, bucket_name, s3_key)
            print(f"✅ Uploaded: {filename} -> s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            print(f"❌ Error uploading {filename}: {e}")
    
    print("-" * 50)
    print("Upload complete!")
    
    # List uploaded files
    print("\nFiles in S3 bucket:")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=s3_prefix)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"   - {obj['Key']} ({obj['Size']} bytes)")


if __name__ == "__main__":
    # CHANGE THIS to your bucket name
    BUCKET_NAME = "sentiment-analysis-demo-arnoldnemeth"
    
    upload_to_s3(bucket_name=BUCKET_NAME)