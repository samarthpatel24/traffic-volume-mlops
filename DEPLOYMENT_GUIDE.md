# MLOps Traffic Volume Predictor - Deployment Guide

## Complete Step-by-Step Guide for AWS Deployment

### Prerequisites
- AWS Account with appropriate permissions
- Docker Hub account
- Git repository (GitHub)
- Local development environment with Python, Docker, and AWS CLI

---

## Phase 1: DVC Setup and Data Management

### 1.1 Initialize DVC in your project
```bash
# Initialize DVC
dvc init

# Add data to DVC tracking
dvc add data/raw/Metro_Interstate_Traffic_Volume.csv

# Commit DVC files
git add data/raw/Metro_Interstate_Traffic_Volume.csv.dvc .dvcignore
git commit -m "Add data to DVC tracking"
```

### 1.2 Set up DVC Remote (S3)
```bash
# Configure S3 bucket as DVC remote
dvc remote add -d myremote s3://your-bucket-name/dvc-cache

# Push data to remote
dvc push
```

### 1.3 Run DVC Pipeline
```bash
# Run the complete pipeline
dvc repro

# Check status
dvc status
```

---

## Phase 2: Local Development and Testing

### 2.1 Test Application Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### 2.2 Test Docker Build Locally
```bash
# Build Docker image
docker build -t traffic-volume-predictor:local .

# Run container locally
docker run -p 8501:8501 traffic-volume-predictor:local

# Test with Docker Compose
docker-compose up --build
```

---

## Phase 3: AWS Infrastructure Setup

### 3.1 Create S3 Bucket for Data Storage
```bash
# Create S3 bucket
aws s3 mb s3://traffic-predictor-data-bucket

# Upload data to S3
aws s3 cp data/raw/Metro_Interstate_Traffic_Volume.csv s3://traffic-predictor-data-bucket/data/
```

### 3.2 Create EC2 Key Pair
```bash
# Create key pair
aws ec2 create-key-pair --key-name traffic-predictor-key --query 'KeyMaterial' --output text > traffic-predictor-key.pem

# Set correct permissions
chmod 400 traffic-predictor-key.pem
```

### 3.3 Launch EC2 Instance
Option A: Using AWS Console
1. Go to EC2 Dashboard
2. Launch Instance
3. Select Amazon Linux 2 AMI
4. Choose t2.micro (free tier)
5. Configure Security Group:
   - SSH (22): Your IP
   - HTTP (80): 0.0.0.0/0
   - Custom TCP (8501): 0.0.0.0/0
6. Select your key pair
7. Launch instance

Option B: Using CloudFormation
```bash
aws cloudformation create-stack --stack-name traffic-predictor-stack --template-body file://aws/cloudformation-template.yaml --parameters ParameterKey=KeyPairName,ParameterValue=traffic-predictor-key
```

---

## Phase 4: Docker Registry Setup

### 4.1 Build and Push to Docker Hub
```bash
# Login to Docker Hub
docker login

# Build image with proper tag
docker build -t yourusername/traffic-volume-predictor:latest .

# Push to Docker Hub
docker push yourusername/traffic-volume-predictor:latest
```

---

## Phase 5: EC2 Instance Configuration

### 5.1 Connect to EC2 Instance
```bash
# SSH into instance
ssh -i traffic-predictor-key.pem ec2-user@YOUR-EC2-PUBLIC-IP
```

### 5.2 Install Docker on EC2
```bash
# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -a -G docker ec2-user

# Logout and login again for group changes to take effect
exit
ssh -i traffic-predictor-key.pem ec2-user@YOUR-EC2-PUBLIC-IP
```

### 5.3 Deploy Application
```bash
# Pull your Docker image
docker pull yourusername/traffic-volume-predictor:latest

# Run the container
docker run -d --name traffic-app --restart unless-stopped -p 80:8501 yourusername/traffic-volume-predictor:latest

# Check if container is running
docker ps
```

---

## Phase 6: CI/CD Pipeline Setup

### 6.1 Configure GitHub Secrets
Go to GitHub repository → Settings → Secrets and add:
- `DOCKER_USERNAME`: Your Docker Hub username
- `DOCKER_PASSWORD`: Your Docker Hub password
- `AWS_HOST`: Your EC2 public IP
- `AWS_USERNAME`: ec2-user
- `AWS_SSH_KEY`: Contents of your .pem file

### 6.2 Push Code to GitHub
```bash
git add .
git commit -m "Add Docker and CI/CD configuration"
git push origin main
```

---

## Phase 7: Testing and Verification

### 7.1 Test Application Access
- Open browser: `http://YOUR-EC2-PUBLIC-IP`
- Verify all features work correctly
- Test predictions with sample data

### 7.2 Verify CI/CD Pipeline
- Make a small change to code
- Push to GitHub
- Check GitHub Actions for pipeline execution
- Verify automatic deployment

---

## Phase 8: Monitoring and Maintenance

### 8.1 Set up Basic Monitoring
```bash
# Check application logs
docker logs traffic-app

# Monitor resource usage
docker stats traffic-app

# Check system resources
htop
df -h
```

### 8.2 Backup Strategy
```bash
# Backup models and data
aws s3 sync models/ s3://traffic-predictor-data-bucket/models/
aws s3 sync metrics/ s3://traffic-predictor-data-bucket/metrics/
```

---

## Troubleshooting Guide

### Common Issues and Solutions

1. **Container won't start**
   ```bash
   docker logs traffic-app
   docker exec -it traffic-app /bin/bash
   ```

2. **Port access issues**
   - Check EC2 Security Groups
   - Verify firewall settings
   - Test with telnet: `telnet YOUR-EC2-IP 80`

3. **Memory issues**
   ```bash
   # Check memory usage
   free -m
   # If needed, create swap space
   sudo dd if=/dev/zero of=/swapfile bs=1M count=1024
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

4. **Docker space issues**
   ```bash
   # Clean up Docker
   docker system prune -a
   ```

---

## Security Best Practices

1. **Update EC2 Security Groups**
   - Restrict SSH access to your IP only
   - Use HTTPS when possible

2. **Update packages regularly**
   ```bash
   sudo yum update -y
   ```

3. **Use IAM roles instead of access keys**

4. **Enable CloudTrail for auditing**

---

## Cost Optimization

1. **Use t2.micro for development (free tier)**
2. **Stop instances when not needed**
3. **Use S3 lifecycle policies for old data**
4. **Monitor AWS costs regularly**

---

## Next Steps

1. **Set up HTTPS with Let's Encrypt**
2. **Implement proper logging and monitoring**
3. **Add health checks and auto-scaling**
4. **Set up database for storing predictions**
5. **Implement user authentication**

---

## Important URLs and Commands Summary

### Key Commands
```bash
# Local testing
streamlit run app.py
docker-compose up

# AWS deployment
docker build -t yourusername/traffic-volume-predictor .
docker push yourusername/traffic-volume-predictor
ssh -i key.pem ec2-user@EC2-IP

# Application access
http://YOUR-EC2-PUBLIC-IP
```

### Important Files Created
- `Dockerfile` - Container configuration
- `.github/workflows/ci-cd.yaml` - CI/CD pipeline
- `docker-compose.yml` - Local development
- `aws/cloudformation-template.yaml` - Infrastructure as Code
- `scripts/deploy.sh` - Deployment automation

Remember to replace placeholders like `yourusername`, `YOUR-EC2-PUBLIC-IP`, and bucket names with your actual values.