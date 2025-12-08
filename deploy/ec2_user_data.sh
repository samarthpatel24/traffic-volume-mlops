#!/bin/bash

# EC2 User Data Script - Runs automatically when EC2 instance starts
# This script will set up everything needed for the Traffic Volume MLOps application

# Log all output to a file for debugging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "🚀 Starting automated EC2 setup for Traffic Volume MLOps..."

# Update system packages
echo "📦 Updating system packages..."
yum update -y

# Install Docker
echo "🐳 Installing Docker..."
yum install -y docker

# Start Docker service
echo "▶️ Starting Docker service..."
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
echo "👥 Adding ec2-user to docker group..."
usermod -a -G docker ec2-user

# Install docker-compose
echo "🔧 Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Create application directory
echo "📁 Creating application directory..."
mkdir -p /home/ec2-user/traffic-volume-mlops
cd /home/ec2-user/traffic-volume-mlops

# Set ownership to ec2-user
chown -R ec2-user:ec2-user /home/ec2-user/traffic-volume-mlops

# Create docker-compose.yml
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
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  default:
    driver: bridge
EOL

# Wait for Docker to be fully ready
echo "⏳ Waiting for Docker to be ready..."
sleep 10

# Pull the Docker image
echo "⬇️ Pulling Docker image from Docker Hub..."
docker pull matrix2415/traffic-volume-mlops:latest

# Start the application
echo "▶️ Starting application..."
/usr/local/bin/docker-compose up -d

# Create systemd service for auto-restart
echo "🔄 Creating systemd service..."
cat > /etc/systemd/system/traffic-volume-mlops.service << 'EOL'
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

# Enable the service
systemctl daemon-reload
systemctl enable traffic-volume-mlops.service

# Create management scripts for ec2-user
echo "📜 Creating management scripts..."

# Start script
cat > /home/ec2-user/start_app.sh << 'EOL'
#!/bin/bash
cd /home/ec2-user/traffic-volume-mlops
docker-compose up -d
echo "✅ Application started!"
echo "🌐 Access at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
EOL

# Stop script
cat > /home/ec2-user/stop_app.sh << 'EOL'
#!/bin/bash
cd /home/ec2-user/traffic-volume-mlops
docker-compose down
echo "✅ Application stopped!"
EOL

# Update script
cat > /home/ec2-user/update_app.sh << 'EOL'
#!/bin/bash
cd /home/ec2-user/traffic-volume-mlops
echo "🔄 Updating application..."
docker-compose down
docker pull matrix2415/traffic-volume-mlops:latest
docker-compose up -d
echo "✅ Application updated!"
EOL

# Logs script
cat > /home/ec2-user/view_logs.sh << 'EOL'
#!/bin/bash
cd /home/ec2-user/traffic-volume-mlops
docker-compose logs -f
EOL

# Make scripts executable and set ownership
chmod +x /home/ec2-user/*.sh
chown ec2-user:ec2-user /home/ec2-user/*.sh

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 30

# Get instance information
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

echo "🎉 Setup completed successfully!"
echo "📊 Instance Information:"
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP: $PUBLIC_IP"
echo "  Private IP: $PRIVATE_IP"
echo ""
echo "🌐 Application URLs:"
echo "  Main: http://$PUBLIC_IP"
echo "  Alt: http://$PUBLIC_IP:8501"
echo ""
echo "📚 Management commands (run as ec2-user):"
echo "  ./start_app.sh   - Start application"
echo "  ./stop_app.sh    - Stop application"
echo "  ./update_app.sh  - Update application"
echo "  ./view_logs.sh   - View application logs"
echo ""
echo "✅ Traffic Volume MLOps is now running!"