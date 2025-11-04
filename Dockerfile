# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models metrics plots

# Copy the data files (if they exist locally)
# Note: In production, you'd pull from S3 or DVC remote
COPY data/ data/

# Expose port for Flask
EXPOSE 8501

# Health check for Flask
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Set environment variables for Flask
ENV FLASK_APP=flask_app.py
ENV FLASK_ENV=production
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlops
RUN chown -R mlops:mlops /app
USER mlops

# Run Flask app
CMD ["python", "flask_app.py"]