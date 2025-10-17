@echo off
REM Windows deployment script for local testing

echo Starting local Docker deployment...

REM Build Docker image
echo Building Docker image...
docker build -t traffic-volume-predictor:latest .

REM Stop existing container if running
echo Stopping existing container...
docker stop traffic-app 2>nul
docker rm traffic-app 2>nul

REM Run new container
echo Starting new container...
docker run -d ^
    --name traffic-app ^
    --restart unless-stopped ^
    -p 8501:8501 ^
    traffic-volume-predictor:latest

echo Deployment completed!
echo Application should be accessible at http://localhost:8501
pause