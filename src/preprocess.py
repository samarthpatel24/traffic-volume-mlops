import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
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

def feature_engineering(df):
    """Create new features from existing ones"""
    # Convert date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
    
    # Extract time-based features
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Create rush hour feature
    df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
    
    # Handle holiday feature (convert None to a proper value)
    df['is_holiday'] = (df['holiday'] != 'None').astype(int)
    
    # Convert temperature from Kelvin to Celsius
    df['temp_celsius'] = df['temp'] - 273.15
    
    # Create weather severity score
    weather_severity_map = {
        'Clear': 1,
        'Clouds': 2,
        'Mist': 3,
        'Rain': 4,
        'Drizzle': 4,
        'Snow': 5,
        'Fog': 3,
        'Haze': 3,
        'Thunderstorm': 5,
        'Smoke': 3
    }
    df['weather_severity'] = df['weather_main'].map(weather_severity_map).fillna(2)
    
    # Create precipitation feature
    df['total_precipitation'] = df['rain_1h'] + df['snow_1h']
    
    print("Feature engineering completed")
    return df

def encode_categorical_features(df):
    """Encode categorical features"""
    # Label encode weather_main and weather_description
    le_weather_main = LabelEncoder()
    le_weather_desc = LabelEncoder()
    
    df['weather_main_encoded'] = le_weather_main.fit_transform(df['weather_main'])
    df['weather_description_encoded'] = le_weather_desc.fit_transform(df['weather_description'])
    
    # Save encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_weather_main, 'models/weather_main_encoder.pkl')
    joblib.dump(le_weather_desc, 'models/weather_description_encoder.pkl')
    
    print("Categorical encoding completed")
    return df

def select_features(df):
    """Select features for modeling"""
    feature_columns = [
        'temp_celsius', 'rain_1h', 'snow_1h', 'clouds_all',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        'is_holiday', 'weather_severity', 'total_precipitation',
        'weather_main_encoded', 'weather_description_encoded'
    ]
    
    target_column = 'traffic_volume'
    
    X = df[feature_columns]
    y = df[target_column]
    
    print(f"Features selected: {X.shape[1]} features")
    return X, y, feature_columns

def scale_features(X_train, X_test):
    """Scale numerical features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, 'models/feature_scaler.pkl')
    
    return X_train_scaled, X_test_scaled, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, feature_columns):
    """Save processed data"""
    os.makedirs('data/processed', exist_ok=True)
    
    # Convert to DataFrames and save
    train_df = pd.DataFrame(X_train, columns=feature_columns)
    train_df['traffic_volume'] = y_train.values
    
    test_df = pd.DataFrame(X_test, columns=feature_columns)
    test_df['traffic_volume'] = y_test.values
    
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    # Save feature names
    with open('data/processed/feature_names.txt', 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    
    print("Processed data saved")

def preprocess_pipeline():
    """Complete preprocessing pipeline"""
    # Load configuration
    config = load_config()
    
    # Load and clean data
    df = load_data(config['data']['raw_path'])
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    df = encode_categorical_features(df)
    
    # Select features and target
    X, y, feature_columns = select_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save processed data
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_columns)
    
    print("Preprocessing pipeline completed successfully!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

if __name__ == "__main__":
    preprocess_pipeline()