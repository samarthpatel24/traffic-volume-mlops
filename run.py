#!/usr/bin/env python3
"""
Setup and run script for the Traffic Volume Prediction MLOps Project
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'streamlit', 
        'plotly', 'seaborn', 'matplotlib', 'dvc', 'joblib', 'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        install_cmd = f"{sys.executable} -m pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing missing packages"):
            return False
    
    print("✅ All required packages are available")
    return True

def setup_project():
    """Setup the project structure and initialize DVC"""
    print("🏗️ Setting up project...")
    
    # Create directories
    directories = ['data/raw', 'data/processed', 'models', 'metrics', 'plots', 'notebooks']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Initialize git if not exists
    if not os.path.exists('.git'):
        if not run_command("git init", "Initializing Git repository"):
            return False
    
    # Initialize DVC if not exists
    if not os.path.exists('.dvc'):
        if not run_command("dvc init", "Initializing DVC"):
            return False
    
    return True

def run_pipeline():
    """Run the complete DVC pipeline"""
    print("🚀 Running DVC pipeline...")
    
    # Run the complete pipeline
    if not run_command("dvc repro", "Running DVC pipeline"):
        print("Pipeline failed. Trying to run stages individually...")
        
        # Try running stages individually
        stages = ['preprocess', 'train', 'evaluate']
        for stage in stages:
            if not run_command(f"dvc repro {stage}", f"Running {stage} stage"):
                print(f"Stage {stage} failed. Please check the logs.")
                return False
    
    print("✅ Pipeline completed successfully!")
    return True

def start_streamlit():
    """Start the Streamlit application"""
    print("🌐 Starting Streamlit application...")
    print("The application will open in your default browser")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")

def main():
    """Main setup and run function"""
    print("🚗 Traffic Volume Prediction MLOps Project Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('config/config.yaml'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        sys.exit(1)
    
    # Step 2: Setup project
    if not setup_project():
        print("❌ Project setup failed")
        sys.exit(1)
    
    # Step 3: Run pipeline
    print("\n" + "=" * 50)
    choice = input("Do you want to run the ML pipeline now? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes']:
        if not run_pipeline():
            print("❌ Pipeline execution failed")
            sys.exit(1)
        
        # Step 4: Start Streamlit app
        print("\n" + "=" * 50)
        choice = input("Do you want to start the Streamlit app now? (y/n): ").lower().strip()
        
        if choice in ['y', 'yes']:
            start_streamlit()
    else:
        print("\n📋 Manual steps to run later:")
        print("1. Run the pipeline: dvc repro")
        print("2. Start the app: streamlit run app.py")
    
    print("\n🎉 Setup completed!")
    print("📖 Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()