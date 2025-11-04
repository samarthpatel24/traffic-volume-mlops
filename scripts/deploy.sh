#!/bin/bash

# AWS deployment script
# This script automates the deployment of the traffic volume predictor to AWS EC2

set -e

# Configuration
DOCKER_IMAGE="matrix2415/traffic-volume-predictor"
EC2_HOST="ec2-34-207-251-181.compute-1.amazonaws.com"
EC2_USER="ec2-user"
SSH_KEY_PATH="samarth_mlops_5.pem"

echo "Starting deployment to AWS EC2..."

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
        echo "Docker installed. Please logout and login again, then rerun this script."
        exit 0
    fi
    
    # Start Docker service if not running
    sudo systemctl start docker
    
    # Pull latest image
    sudo docker pull matrix2415/traffic-volume-predictor:latest
    
    # Stop existing container
    sudo docker stop traffic-app 2>/dev/null || true
    sudo docker rm traffic-app 2>/dev/null || true
    
    # Run new container
    sudo docker run -d \
        --name traffic-app \
        --restart unless-stopped \
        -p 80:8501 \
        matrix2415/traffic-volume-predictor:latest
    
    echo "Deployment completed successfully!"
    echo "Application should be accessible at http://ec2-34-207-251-181.compute-1.amazonaws.com"
EOF

echo "Deployment script completed!"