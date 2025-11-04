# AWS deployment script for PowerShell
# This script automates the deployment of the traffic volume predictor to AWS EC2

param(
    [string]$DockerImage = "matrix2415/traffic-volume-predictor",
    [string]$EC2Host = "ec2-34-207-251-181.compute-1.amazonaws.com",
    [string]$EC2User = "ec2-user",
    [string]$SSHKeyPath = "samarth_mlops_5.pem"
)

Write-Host "Starting deployment to AWS EC2..." -ForegroundColor Green

# Check if SSH key exists
if (-not (Test-Path $SSHKeyPath)) {
    Write-Host "SSH key file not found: $SSHKeyPath" -ForegroundColor Red
    Write-Host "Please make sure the key file is in the current directory" -ForegroundColor Yellow
    exit 1
}

# Deploy to EC2
Write-Host "Connecting to EC2 instance and deploying..." -ForegroundColor Yellow

$deployCommands = @"
# Update system
sudo yum update -y

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo yum install -y docker
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -a -G docker ec2-user
    echo "Docker installed. Logging out and back in..."
    exec su -l ec2-user
fi

# Ensure Docker is running
sudo systemctl start docker

# Pull latest image
echo "Pulling latest Docker image..."
sudo docker pull $DockerImage:latest

# Stop existing container
echo "Stopping existing container..."
sudo docker stop traffic-app 2>/dev/null || true
sudo docker rm traffic-app 2>/dev/null || true

# Run new container
echo "Starting new container..."
sudo docker run -d \
    --name traffic-app \
    --restart unless-stopped \
    -p 80:8501 \
    $DockerImage:latest

# Check if container is running
echo "Checking container status..."
sudo docker ps | grep traffic-app

echo "Deployment completed successfully!"
echo "Application should be accessible at http://$EC2Host"
"@

# Execute deployment commands on EC2
ssh -i $SSHKeyPath "$EC2User@$EC2Host" $deployCommands

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment completed successfully!" -ForegroundColor Green
    Write-Host "Application URL: http://$EC2Host" -ForegroundColor Cyan
} else {
    Write-Host "Deployment failed. Check the error messages above." -ForegroundColor Red
}