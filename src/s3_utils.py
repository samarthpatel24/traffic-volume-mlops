"""
S3 utilities for MLOps model and data management
"""

import boto3
import os
import json
import joblib
from datetime import datetime
from botocore.exceptions import NoCredentialsError, ClientError

class S3ModelManager:
    """Manages model and data storage in AWS S3"""
    
    def __init__(self, bucket_name=None, region='us-east-1'):
        """
        Initialize S3 client
        
        Args:
            bucket_name (str): S3 bucket name
            region (str): AWS region
        """
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'traffic-mlops-bucket')
        self.region = region
        
        try:
            self.s3_client = boto3.client('s3', region_name=region)
            self.s3_resource = boto3.resource('s3', region_name=region)
            print(f"✅ S3 client initialized for bucket: {self.bucket_name}")
        except NoCredentialsError:
            print("❌ AWS credentials not found")
            self.s3_client = None
            self.s3_resource = None
    
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            print(f"✅ Bucket {self.bucket_name} created successfully")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'BucketAlreadyOwnedByYou':
                print(f"✅ Bucket {self.bucket_name} already exists")
            else:
                print(f"❌ Error creating bucket: {e}")
    
    def upload_file(self, local_path, s3_key):
        """
        Upload file to S3
        
        Args:
            local_path (str): Local file path
            s3_key (str): S3 object key
            
        Returns:
            bool: Success status
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            print(f"✅ Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"❌ Error uploading {local_path}: {e}")
            return False
    
    def download_file(self, s3_key, local_path):
        """
        Download file from S3
        
        Args:
            s3_key (str): S3 object key
            local_path (str): Local file path
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            print(f"✅ Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
        except Exception as e:
            print(f"❌ Error downloading {s3_key}: {e}")
            return False
    
    def file_exists(self, s3_key):
        """
        Check if file exists in S3
        
        Args:
            s3_key (str): S3 object key
            
        Returns:
            bool: File existence status
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False
    
    def list_models(self, prefix='models/'):
        """
        List all models in S3
        
        Args:
            prefix (str): S3 key prefix
            
        Returns:
            list: List of model files
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            models = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.pkl'):
                        models.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat()
                        })
            
            return sorted(models, key=lambda x: x['last_modified'], reverse=True)
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def upload_model(self, model, model_name, model_info=None):
        """
        Upload trained model to S3
        
        Args:
            model: Trained model object
            model_name (str): Model name/identifier
            model_info (dict): Model metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Save model locally first
            local_model_path = f"temp_{model_name}.pkl"
            joblib.dump(model, local_model_path)
            
            # Upload model to S3
            s3_model_key = f"models/{model_name}.pkl"
            success = self.upload_file(local_model_path, s3_model_key)
            
            # Upload model info if provided
            if model_info and success:
                local_info_path = f"temp_{model_name}_info.json"
                with open(local_info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                s3_info_key = f"models/{model_name}_info.json"
                self.upload_file(local_info_path, s3_info_key)
                
                # Cleanup local temp files
                os.remove(local_info_path)
            
            # Cleanup local temp files
            os.remove(local_model_path)
            
            return success
        except Exception as e:
            print(f"❌ Error uploading model {model_name}: {e}")
            return False
    
    def download_model(self, model_name, local_dir='models'):
        """
        Download model from S3
        
        Args:
            model_name (str): Model name/identifier
            local_dir (str): Local directory to save model
            
        Returns:
            tuple: (model_object, model_info_dict) or (None, None)
        """
        try:
            # Download model file
            s3_model_key = f"models/{model_name}.pkl"
            local_model_path = os.path.join(local_dir, f"{model_name}.pkl")
            
            if not self.download_file(s3_model_key, local_model_path):
                return None, None
            
            # Load model
            model = joblib.load(local_model_path)
            
            # Download model info if exists
            model_info = None
            s3_info_key = f"models/{model_name}_info.json"
            local_info_path = os.path.join(local_dir, f"{model_name}_info.json")
            
            if self.file_exists(s3_info_key):
                if self.download_file(s3_info_key, local_info_path):
                    with open(local_info_path, 'r') as f:
                        model_info = json.load(f)
            
            return model, model_info
        except Exception as e:
            print(f"❌ Error downloading model {model_name}: {e}")
            return None, None
    
    def get_latest_model(self):
        """
        Get the latest model from S3
        
        Returns:
            dict: Latest model information
        """
        models = self.list_models()
        if models:
            return models[0]  # Already sorted by last_modified desc
        return None
    
    def sync_local_to_s3(self, local_dir, s3_prefix):
        """
        Sync local directory to S3
        
        Args:
            local_dir (str): Local directory path
            s3_prefix (str): S3 prefix
        """
        try:
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, local_dir)
                    s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                    
                    self.upload_file(local_path, s3_key)
                    
            print(f"✅ Synced {local_dir} to s3://{self.bucket_name}/{s3_prefix}")
        except Exception as e:
            print(f"❌ Error syncing {local_dir}: {e}")

# Utility functions for easy access
def create_s3_manager():
    """Create S3 manager instance"""
    return S3ModelManager()

def upload_current_models_to_s3():
    """Upload all current models to S3"""
    s3_manager = create_s3_manager()
    s3_manager.create_bucket_if_not_exists()
    
    # Upload models directory
    if os.path.exists('models'):
        s3_manager.sync_local_to_s3('models', 'models')
    
    # Upload data directory
    if os.path.exists('data'):
        s3_manager.sync_local_to_s3('data', 'data')
    
    print("✅ All models and data uploaded to S3")

if __name__ == "__main__":
    # Test S3 connection and upload current models
    upload_current_models_to_s3()