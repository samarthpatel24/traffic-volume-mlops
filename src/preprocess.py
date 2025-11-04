import pandas as pd
import numpy as np
from datetime import datetime
import os
import yaml

def load_config(config_path='config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(data_path):
    """Load the raw dataset"""
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    return df

def clean_data(df):
    """Clean the dataset"""
    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())
    
    # Handle missing values in holiday column (most are None, which is valid)
    # We don't drop rows with None in holiday as it means "no holiday"
    
    # Only drop rows where critical columns have missing values
    critical_columns = ['traffic_volume', 'temp', 'weather_main', 'weather_description', 'date_time']
    df = df.dropna(subset=critical_columns)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"Dataset after cleaning: {df.shape}")
    return df

def preprocess_pipeline():
    """Data preprocessing pipeline - cleaning only"""
    # Load configuration
    config = load_config()
    
    # Load and clean data
    df = load_data(config['data']['raw_path'])
    df = clean_data(df)
    
    # Save cleaned data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_data.csv', index=False)
    
    print("Data preprocessing (cleaning) completed successfully!")
    return df

if __name__ == "__main__":
    preprocess_pipeline()