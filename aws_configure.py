#!/usr/bin/env python3
"""
AWS Configuration Test Script
This script tests AWS configuration using credentials from .env file
"""

import os
import sys
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

def load_env_credentials():
    """Load AWS credentials from .env file"""
    # Check if .env file exists
    env_path = os.path.join(os.getcwd(), '.env')
    if not os.path.exists(env_path):
        print(f"⚠️  .env file not found at: {env_path}")
        return None
    
    print(f"📁 Loading .env file from: {env_path}")
    
    # Try loading with python-dotenv first
    load_dotenv(env_path)
    
    # Manual fallback - read and parse the file
    env_vars = {}
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📄 .env file content preview (first 200 chars):")
            print(repr(content[:200]))
            
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    env_vars[key] = value
                    if key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']:
                        print(f"📋 Line {line_num}: {key} = {value[:4]}...{value[-4:] if len(value) > 8 else value}")
    except Exception as e:
        print(f"⚠️  Error reading .env file: {e}")
    
    print(f"🔍 Parsed env vars: {list(env_vars.keys())}")
    
    # Try from environment first, then from manual parsing
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID') or env_vars.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY') or env_vars.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION') or env_vars.get('AWS_DEFAULT_REGION', 'us-east-1')
    s3_bucket = os.getenv('S3_BUCKET_NAME') or env_vars.get('S3_BUCKET_NAME')
    ecr_repository = os.getenv('ECR_REPOSITORY') or env_vars.get('ECR_REPOSITORY')
    aws_account_id = os.getenv('AWS_ACCOUNT_ID') or env_vars.get('AWS_ACCOUNT_ID')
    
    print(f"🔍 Debug - Access Key found: {bool(aws_access_key)} (length: {len(aws_access_key) if aws_access_key else 0})")
    print(f"🔍 Debug - Secret Key found: {bool(aws_secret_key)} (length: {len(aws_secret_key) if aws_secret_key else 0})")
    
    return {
        'access_key': aws_access_key,
        'secret_key': aws_secret_key,
        'region': aws_region,
        's3_bucket': s3_bucket,
        'ecr_repository': ecr_repository,
        'account_id': aws_account_id
    }

def test_credentials_format(credentials):
    """Test if credentials are properly formatted"""
    print("🔍 Testing credential format...")
    
    issues = []
    
    if not credentials['access_key']:
        issues.append("❌ AWS_ACCESS_KEY_ID is not set")
    elif len(credentials['access_key']) != 20:
        issues.append(f"⚠️  AWS_ACCESS_KEY_ID length is {len(credentials['access_key'])}, expected 20")
    else:
        print(f"✅ AWS_ACCESS_KEY_ID format looks good (starts with {credentials['access_key'][:4]}...)")
    
    if not credentials['secret_key']:
        issues.append("❌ AWS_SECRET_ACCESS_KEY is not set")
    elif len(credentials['secret_key']) != 40:
        issues.append(f"⚠️  AWS_SECRET_ACCESS_KEY length is {len(credentials['secret_key'])}, expected 40")
    else:
        print(f"✅ AWS_SECRET_ACCESS_KEY format looks good (starts with {credentials['secret_key'][:4]}...)")
    
    if not credentials['region']:
        issues.append("❌ AWS_DEFAULT_REGION is not set")
    else:
        print(f"✅ AWS region: {credentials['region']}")
    
    if issues:
        print("\n".join(issues))
        return False
    
    return True

def test_aws_connection(credentials):
    """Test AWS connection using STS (Security Token Service)"""
    print("\n🔍 Testing AWS connection...")
    
    try:
        # Create STS client to test credentials
        sts_client = boto3.client(
            'sts',
            aws_access_key_id=credentials['access_key'],
            aws_secret_access_key=credentials['secret_key'],
            region_name=credentials['region']
        )
        
        # Get caller identity
        response = sts_client.get_caller_identity()
        
        print(f"✅ AWS connection successful!")
        print(f"   Account ID: {response.get('Account')}")
        print(f"   User ARN: {response.get('Arn')}")
        print(f"   User ID: {response.get('UserId')}")
        
        return True
        
    except NoCredentialsError:
        print("❌ AWS credentials not found or invalid")
        return False
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'InvalidUserID.NotFound':
            print("❌ AWS Access Key ID not found")
        elif error_code == 'SignatureDoesNotMatch':
            print("❌ AWS Secret Access Key is incorrect")
        elif error_code == 'AccessDenied':
            print("❌ Access denied - check IAM permissions")
        else:
            print(f"❌ AWS connection failed: {error_code} - {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_s3_access(credentials):
    """Test S3 bucket access"""
    print("\n🔍 Testing S3 access...")
    
    if not credentials['s3_bucket']:
        print("⚠️  S3_BUCKET_NAME not configured in .env")
        return False
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=credentials['access_key'],
            aws_secret_access_key=credentials['secret_key'],
            region_name=credentials['region']
        )
        
        # Test bucket access
        response = s3_client.head_bucket(Bucket=credentials['s3_bucket'])
        print(f"✅ S3 bucket '{credentials['s3_bucket']}' is accessible")
        
        # List some objects to test read access
        try:
            objects = s3_client.list_objects_v2(
                Bucket=credentials['s3_bucket'],
                MaxKeys=5
            )
            object_count = objects.get('KeyCount', 0)
            print(f"✅ S3 read access confirmed - found {object_count} objects")
            
            if object_count > 0:
                print("   Sample objects:")
                for obj in objects.get('Contents', [])[:3]:
                    print(f"   - {obj['Key']} ({obj['Size']} bytes)")
                    
        except ClientError as e:
            print(f"⚠️  S3 list objects failed: {e.response['Error']['Message']}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchBucket':
            print(f"❌ S3 bucket '{credentials['s3_bucket']}' does not exist")
        elif error_code == 'AccessDenied':
            print(f"❌ Access denied to S3 bucket '{credentials['s3_bucket']}'")
        else:
            print(f"❌ S3 access failed: {error_code} - {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"❌ S3 test failed: {str(e)}")
        return False

def test_ecr_access(credentials):
    """Test ECR repository access"""
    print("\n🔍 Testing ECR access...")
    
    if not credentials['ecr_repository']:
        print("⚠️  ECR_REPOSITORY not configured in .env")
        return False
    
    try:
        ecr_client = boto3.client(
            'ecr',
            aws_access_key_id=credentials['access_key'],
            aws_secret_access_key=credentials['secret_key'],
            region_name=credentials['region']
        )
        
        # Test ECR repository access
        response = ecr_client.describe_repositories(
            repositoryNames=[credentials['ecr_repository']]
        )
        
        repo_info = response['repositories'][0]
        print(f"✅ ECR repository '{credentials['ecr_repository']}' is accessible")
        print(f"   Repository URI: {repo_info['repositoryUri']}")
        print(f"   Created: {repo_info['createdAt']}")
        
        # Test if we can get auth token
        try:
            auth_response = ecr_client.get_authorization_token()
            print("✅ ECR authorization token retrieved successfully")
        except ClientError as e:
            print(f"⚠️  ECR auth token failed: {e.response['Error']['Message']}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'RepositoryNotFoundException':
            print(f"❌ ECR repository '{credentials['ecr_repository']}' does not exist")
        elif error_code == 'AccessDenied':
            print(f"❌ Access denied to ECR repository '{credentials['ecr_repository']}'")
        else:
            print(f"❌ ECR access failed: {error_code} - {e.response['Error']['Message']}")
        return False
    except Exception as e:
        print(f"❌ ECR test failed: {str(e)}")
        return False

def test_iam_permissions(credentials):
    """Test basic IAM permissions"""
    print("\n🔍 Testing IAM permissions...")
    
    try:
        iam_client = boto3.client(
            'iam',
            aws_access_key_id=credentials['access_key'],
            aws_secret_access_key=credentials['secret_key'],
            region_name=credentials['region']
        )
        
        # Test get user (works for IAM users)
        try:
            response = iam_client.get_user()
            user_name = response['User']['UserName']
            print(f"✅ IAM user access confirmed - User: {user_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'AccessDenied':
                print("⚠️  Limited IAM access (might be using temporary credentials)")
            else:
                print(f"⚠️  IAM get_user failed: {e.response['Error']['Message']}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  IAM test failed: {str(e)}")
        return False

def main():
    """Main function to run all AWS configuration tests"""
    print("🔧 AWS Configuration Test")
    print("=" * 50)
    
    # Load credentials from .env
    credentials = load_env_credentials()
    
    if credentials is None:
        print("\n❌ Failed to load .env file. Please ensure .env exists in the current directory.")
        return False
    
    # Test credential format
    if not test_credentials_format(credentials):
        print("\n❌ Credential format issues found. Please check your .env file.")
        return False
    
    # Test AWS connection
    if not test_aws_connection(credentials):
        print("\n❌ AWS connection failed. Please check your credentials.")
        return False
    
    # Test individual services
    s3_ok = test_s3_access(credentials)
    ecr_ok = test_ecr_access(credentials)
    iam_ok = test_iam_permissions(credentials)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   AWS Connection: ✅")
    print(f"   S3 Access: {'✅' if s3_ok else '❌'}")
    print(f"   ECR Access: {'✅' if ecr_ok else '❌'}")
    print(f"   IAM Access: {'✅' if iam_ok else '⚠️'}")
    
    if s3_ok and ecr_ok:
        print("\n🎉 AWS configuration is working properly!")
        print("   You can now use GitHub Actions CI/CD with these credentials.")
    else:
        print("\n⚠️  Some AWS services have access issues.")
        print("   Please check your IAM permissions.")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)