# EC2 Deployment Guide for Traffic Volume MLOps

This guide provides multiple methods to deploy the Traffic Volume MLOps application on AWS EC2 using Docker Hub.

## 🎯 Overview

- **Docker Image**: `matrix2415/traffic-volume-mlops:latest` (hosted on Docker Hub)
- **Application Port**: 8501 (mapped to port 80 on host)
- **Instance Type**: t3.medium (2 vCPUs, 4GB RAM)
- **Cost**: ~$30/month (much cheaper than ECR)

## 📋 Prerequisites

1. **AWS Account** with EC2 access
2. **AWS CLI** configured with appropriate permissions
3. **SSH Key Pair** created in your AWS region
4. **Terraform** installed (for automated deployment)

## 🚀 Deployment Methods

### Method 1: Automated Deployment with Terraform (Recommended)

This method creates everything automatically including VPC, security groups, and EC2 instance.

#### Step 1: Prepare Terraform Configuration

```bash
cd deploy/
```

#### Step 2: Update Variables

Edit `main.tf` and update these variables:
```hcl
variable "key_name" {
  default = "your-key-pair-name"  # Replace with your key pair name
}

variable "aws_region" {
  default = "us-east-1"  # Change if needed
}
```

#### Step 3: Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Plan the deployment
terraform plan

# Apply the configuration
terraform apply
```

#### Step 4: Access Your Application

After deployment completes, Terraform will output:
- **Instance ID**: The EC2 instance identifier
- **Public IP**: The IP address to access your application
- **Application URLs**: Direct links to your application

```bash
# Example output:
application_url = "http://54.123.45.67"
application_url_alt = "http://54.123.45.67:8501"
ssh_command = "ssh -i ~/.ssh/your-key.pem ec2-user@54.123.45.67"
```

### Method 2: Manual EC2 Setup

If you prefer manual setup or already have an EC2 instance:

#### Step 1: Launch EC2 Instance

1. **Launch Instance**: Amazon Linux 2 AMI
2. **Instance Type**: t3.medium or larger
3. **Security Group**: Allow ports 22 (SSH), 80 (HTTP), 8501 (Flask)
4. **Storage**: 20GB GP3
5. **Key Pair**: Attach your SSH key

#### Step 2: Connect and Setup

```bash
# Connect to your instance
ssh -i ~/.ssh/your-key.pem ec2-user@YOUR_INSTANCE_IP

# Copy and run the setup script
curl -O https://raw.githubusercontent.com/YOUR_USERNAME/traffic-volume-mlops/main/deploy/ec2_setup.sh
chmod +x ec2_setup.sh
./ec2_setup.sh
```

#### Step 3: Start the Application

```bash
# Start the application
./start_app.sh

# View logs
./view_logs.sh
```

### Method 3: Using User Data Script

For automatic setup during instance launch:

#### Step 1: Copy User Data Script

When launching EC2 instance, paste the contents of `ec2_user_data.sh` into the **User Data** field.

#### Step 2: Launch Instance

The application will be automatically installed and started when the instance boots.

## 🔧 Application Management

Once deployed, you can manage the application using these commands:

```bash
# SSH to your instance
ssh -i ~/.ssh/your-key.pem ec2-user@YOUR_INSTANCE_IP

# Start application
./start_app.sh

# Stop application
./stop_app.sh

# Update to latest version
./update_app.sh

# View real-time logs
./view_logs.sh

# Check application status
cd /home/ec2-user/traffic-volume-mlops
docker-compose ps

# Restart application
docker-compose restart
```

## 🌐 Access URLs

Your application will be available at:

- **Primary**: `http://YOUR_INSTANCE_IP` (port 80)
- **Direct**: `http://YOUR_INSTANCE_IP:8501` (port 8501)

## 📊 Monitoring and Logs

### View Application Logs
```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail 100

# System logs
sudo tail -f /var/log/user-data.log
```

### Check Application Health
```bash
# Health check endpoint
curl http://localhost:8501/health

# Container status
docker ps

# Resource usage
docker stats
```

## 🔒 Security Considerations

### Production Security (Important!)

1. **Restrict Access**: Update security group to allow access only from specific IP ranges
```bash
# Example: Allow only your office IP
terraform apply -var='allowed_cidr_blocks=["203.0.113.0/24"]'
```

2. **Enable HTTPS**: Use a load balancer with SSL certificate for production
3. **Update System**: Keep the system updated
```bash
sudo yum update -y
```

4. **Monitor Access**: Enable CloudTrail and VPC Flow Logs

### Environment Variables

For production, set these environment variables:
```bash
# In your docker-compose.yml or .env file
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
FLASK_SECRET_KEY=your_secret_key
```

## 🔄 Continuous Deployment

### Automated Updates

Set up a webhook or GitHub Actions to automatically update when you push new code:

```bash
# Create update script with webhook
cat > /home/ec2-user/webhook_update.sh << 'EOF'
#!/bin/bash
cd /home/ec2-user/traffic-volume-mlops
git pull origin main  # if you want to pull latest configs
docker pull matrix2415/traffic-volume-mlops:latest
docker-compose down
docker-compose up -d
EOF

chmod +x /home/ec2-user/webhook_update.sh
```

### GitHub Actions Integration

Add this to your `.github/workflows/deploy.yml`:

```yaml
- name: Trigger EC2 Update
  run: |
    ssh -i ${{ secrets.EC2_SSH_KEY }} ec2-user@${{ secrets.EC2_HOST }} './webhook_update.sh'
```

## 💰 Cost Optimization

- **Instance Scheduling**: Use AWS Instance Scheduler to stop instance during off-hours
- **Spot Instances**: Use spot instances for development (50-70% cost savings)
- **Right-sizing**: Monitor CPU/memory usage and adjust instance type

## 🆘 Troubleshooting

### Common Issues

1. **Application not accessible**:
   ```bash
   # Check security group allows ports 80 and 8501
   # Check if Docker is running
   sudo systemctl status docker
   
   # Check container status
   docker ps
   ```

2. **Docker permission denied**:
   ```bash
   sudo usermod -a -G docker ec2-user
   # Logout and login again
   ```

3. **Application crashes**:
   ```bash
   # Check logs
   docker-compose logs
   
   # Check system resources
   free -h
   df -h
   ```

4. **Port conflicts**:
   ```bash
   # Check what's using port 80
   sudo netstat -tlnp | grep :80
   ```

## 📞 Support

- **Logs Location**: `/var/log/user-data.log`, `docker-compose logs`
- **Config Location**: `/home/ec2-user/traffic-volume-mlops/`
- **Service Status**: `sudo systemctl status traffic-volume-mlops`

## 🎉 Success Checklist

- [ ] EC2 instance launched successfully
- [ ] Docker installed and running
- [ ] Application container started
- [ ] Security group configured (ports 22, 80, 8501)
- [ ] Application accessible via web browser
- [ ] Health check endpoint responding: `/health`
- [ ] Model prediction working: `/predict`
- [ ] Model management interface working: `/models`

Your Traffic Volume MLOps application is now live on AWS EC2! 🚀