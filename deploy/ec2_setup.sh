#!/bin/bash

# EC2 Deployment Script for Traffic Volume MLOps
# This script sets up Docker and runs the application on EC2

set -e  # Exit on any error

echo "🚀 Starting Traffic Volume MLOps EC2 Deployment..."

# Update system packages
echo "📦 Updating system packages..."
sudo yum update -y

# Install Docker
echo "🐳 Installing Docker..."
sudo yum install -y docker

# Start Docker service
echo "▶️ Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add ec2-user to docker group (so we don't need sudo for docker commands)
echo "👥 Adding user to docker group..."
sudo usermod -a -G docker ec2-user

# Install docker-compose (optional, for future use)
echo "🔧 Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
echo "📁 Creating application directory..."
mkdir -p /home/ec2-user/traffic-volume-mlops
cd /home/ec2-user/traffic-volume-mlops

# Pull the Docker image from Docker Hub
echo "⬇️ Pulling Docker image from Docker Hub..."
# Note: We need to logout and login again for docker group to take effect
newgrp docker << EOF
docker pull matrix2415/traffic-volume-mlops:latest
EOF

# Create environment file for production
echo "⚙️ Creating environment configuration..."
cat > .env << 'EOL'
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
FLASK_HOST=0.0.0.0
FLASK_PORT=8501

# AWS Configuration (if using S3)
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
# AWS_DEFAULT_REGION=us-east-1

# DVC Configuration
DVC_REMOTE_URL=s3://traffic-volume-mlops-bucket/models
EOL

# Create docker-compose.yml for easy management
echo "📝 Creating docker-compose configuration..."
cat > docker-compose.yml << 'EOL'
version: '3.8'

services:
  traffic-volume-app:
    image: matrix2415/traffic-volume-mlops:latest
    container_name: traffic-volume-mlops
    ports:
      - "80:8501"  # Map container port 8501 to host port 80
      - "8501:8501"  # Also expose on 8501 for direct access
    environment:
      - FLASK_ENV=production
      - FLASK_DEBUG=False
    volumes:
      - ./data:/app/data  # Mount data directory (if needed)
      - ./models:/app/models  # Mount models directory (if needed)
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - mlops-network

networks:
  mlops-network:
    driver: bridge
EOL

# Create systemd service for auto-start
echo "🔄 Creating systemd service..."
sudo tee /etc/systemd/system/traffic-volume-mlops.service > /dev/null << 'EOL'
[Unit]
Description=Traffic Volume MLOps Docker Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ec2-user/traffic-volume-mlops
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
User=ec2-user
Group=docker

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd and enable the service
echo "🔄 Enabling auto-start service..."
sudo systemctl daemon-reload
sudo systemctl enable traffic-volume-mlops.service

# Create startup script
echo "📜 Creating startup script..."
cat > start_app.sh << 'EOL'
#!/bin/bash
echo "🚀 Starting Traffic Volume MLOps Application..."

# Ensure Docker is running
sudo systemctl start docker

# Pull latest image (optional)
echo "⬇️ Pulling latest image..."
docker pull matrix2415/traffic-volume-mlops:latest

# Start the application
echo "▶️ Starting application..."
docker-compose up -d

# Show status
echo "📊 Application status:"
docker-compose ps

echo "✅ Application started successfully!"
echo "🌐 Access the application at:"
echo "   - http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "   - http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501"
EOL

chmod +x start_app.sh

# Create stop script
echo "📜 Creating stop script..."
cat > stop_app.sh << 'EOL'
#!/bin/bash
echo "🛑 Stopping Traffic Volume MLOps Application..."
docker-compose down
echo "✅ Application stopped successfully!"
EOL

chmod +x stop_app.sh

# Create update script
echo "📜 Creating update script..."
cat > update_app.sh << 'EOL'
#!/bin/bash
echo "🔄 Updating Traffic Volume MLOps Application..."

# Stop current application
docker-compose down

# Pull latest image
echo "⬇️ Pulling latest image..."
docker pull matrix2415/traffic-volume-mlops:latest

# Remove old containers and images
echo "🧹 Cleaning up old containers..."
docker system prune -f

# Start updated application
echo "▶️ Starting updated application..."
docker-compose up -d

# Show status
echo "📊 Application status:"
docker-compose ps

echo "✅ Application updated successfully!"
EOL

chmod +x update_app.sh

echo "🎉 EC2 setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Run: ./start_app.sh"
echo "2. Configure security group to allow inbound traffic on ports 80 and 8501"
echo "3. Access your application at: http://YOUR_EC2_PUBLIC_IP"
echo ""
echo "📚 Available commands:"
echo "  ./start_app.sh   - Start the application"
echo "  ./stop_app.sh    - Stop the application"
echo "  ./update_app.sh  - Update to latest version"
echo "  docker-compose logs -f  - View application logs"