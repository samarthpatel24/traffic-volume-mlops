#!/bin/bash

# AWS deployment script
# This script automates the deployment of the traffic volume predictor to AWS EC2

set -e

# Configuration
DOCKER_IMAGE="your-dockerhub-username/traffic-volume-predictor"
EC2_HOST="your-ec2-public-ip"
EC2_USER="ec2-user"
SSH_KEY_PATH="path/to/your/key.pem"

echo "Starting deployment to AWS EC2..."

# Build and push Docker image
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE:latest .

echo "Pushing to Docker Hub..."
docker push $DOCKER_IMAGE:latest

# Deploy to EC2
echo "Deploying to EC2 instance..."
ssh -i $SSH_KEY_PATH $EC2_USER@$EC2_HOST << 'EOF'
    # Update system
    sudo yum update -y
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        sudo yum install -y docker
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -a -G docker ec2-user
    fi
    
    # Pull latest image
    sudo docker pull your-dockerhub-username/traffic-volume-predictor:latest
    
    # Stop existing container
    sudo docker stop traffic-app 2>/dev/null || true
    sudo docker rm traffic-app 2>/dev/null || true
    
    # Run new container
    sudo docker run -d \
        --name traffic-app \
        --restart unless-stopped \
        -p 80:8501 \
        your-dockerhub-username/traffic-volume-predictor:latest
    
    echo "Deployment completed successfully!"
    echo "Application should be accessible at http://$EC2_HOST"
EOF

echo "Deployment script completed!"