# Terraform configuration for EC2 deployment
# This will create an EC2 instance with the Traffic Volume MLOps application

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure AWS Provider
provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"  # 2 vCPUs, 4GB RAM - good for ML workloads
}

variable "key_name" {
  description = "AWS Key Pair name for EC2 access"
  type        = string
  default     = ""  # Set this to your key pair name
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Allow from anywhere (change for production)
}

# Data sources
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amzn2-ami-hvm-*-x86_64-gp2"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

# Create VPC
resource "aws_vpc" "mlops_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "traffic-volume-mlops-vpc"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# Create Internet Gateway
resource "aws_internet_gateway" "mlops_igw" {
  vpc_id = aws_vpc.mlops_vpc.id

  tags = {
    Name        = "traffic-volume-mlops-igw"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# Create public subnet
resource "aws_subnet" "mlops_public_subnet" {
  vpc_id                  = aws_vpc.mlops_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name        = "traffic-volume-mlops-public-subnet"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# Create route table
resource "aws_route_table" "mlops_public_rt" {
  vpc_id = aws_vpc.mlops_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mlops_igw.id
  }

  tags = {
    Name        = "traffic-volume-mlops-public-rt"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# Associate route table with subnet
resource "aws_route_table_association" "mlops_public_rta" {
  subnet_id      = aws_subnet.mlops_public_subnet.id
  route_table_id = aws_route_table.mlops_public_rt.id
}

# Security Group for the application
resource "aws_security_group" "mlops_sg" {
  name_prefix = "traffic-volume-mlops-"
  vpc_id      = aws_vpc.mlops_vpc.id

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # HTTP access (port 80)
  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Flask application (port 8501)
  ingress {
    description = "Flask App"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "traffic-volume-mlops-sg"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# IAM role for EC2 instance (if needed for S3 access)
resource "aws_iam_role" "mlops_ec2_role" {
  name = "traffic-volume-mlops-ec2-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "traffic-volume-mlops-ec2-role"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# IAM policy for S3 access (optional)
resource "aws_iam_role_policy" "mlops_s3_policy" {
  name = "traffic-volume-mlops-s3-policy"
  role = aws_iam_role.mlops_ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::traffic-volume-mlops-bucket",
          "arn:aws:s3:::traffic-volume-mlops-bucket/*"
        ]
      }
    ]
  })
}

# IAM instance profile
resource "aws_iam_instance_profile" "mlops_profile" {
  name = "traffic-volume-mlops-profile"
  role = aws_iam_role.mlops_ec2_role.name

  tags = {
    Name        = "traffic-volume-mlops-profile"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# EC2 Instance
resource "aws_instance" "mlops_instance" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.mlops_sg.id]
  subnet_id              = aws_subnet.mlops_public_subnet.id
  iam_instance_profile   = aws_iam_instance_profile.mlops_profile.name

  # User data script for automatic setup
  user_data = file("${path.module}/ec2_user_data.sh")

  # Root volume configuration
  root_block_device {
    volume_type = "gp3"
    volume_size = 20  # 20GB should be sufficient
    encrypted   = true
  }

  tags = {
    Name        = "traffic-volume-mlops-server"
    Project     = "TrafficVolumeMLOps"
    Environment = "production"
  }
}

# Outputs
output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.mlops_instance.id
}

output "instance_public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.mlops_instance.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.mlops_instance.public_dns
}

output "application_url" {
  description = "URL to access the Traffic Volume MLOps application"
  value       = "http://${aws_instance.mlops_instance.public_ip}"
}

output "application_url_alt" {
  description = "Alternative URL to access the application"
  value       = "http://${aws_instance.mlops_instance.public_ip}:8501"
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ec2-user@${aws_instance.mlops_instance.public_ip}"
}