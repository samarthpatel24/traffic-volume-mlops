import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import yaml
import os
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path='config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_processed_data():
    """Load preprocessed training data"""
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Separate features and target
    X_train = train_df.drop('traffic_volume', axis=1)
    y_train = train_df['traffic_volume']
    X_test = test_df.drop('traffic_volume', axis=1)
    y_test = test_df['traffic_volume']
    
    return X_train, X_test, y_train, y_test

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mape': calculate_mape(y_test, y_pred)
    }
    
    return metrics, y_pred

def train_random_forest(X_train, y_train, config):
    """Train Random Forest model"""
    logger = logging.getLogger(__name__)
    logger.info("Training Random Forest model...")
    
    rf_params = config['model']['random_forest']
    
    model = RandomForestRegressor(
        n_estimators=rf_params['n_estimators'],
        max_depth=rf_params['max_depth'],
        min_samples_split=rf_params['min_samples_split'],
        min_samples_leaf=rf_params['min_samples_leaf'],
        random_state=config['model']['random_state'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, 
                               cv=config['training']['cv_folds'], 
                               scoring='neg_mean_squared_error')
    
    logger.info(f"Random Forest CV RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(cv_scores.std() * 2):.2f})")
    
    return model

def train_xgboost(X_train, y_train, config):
    """Train XGBoost model"""
    logger = logging.getLogger(__name__)
    logger.info("Training XGBoost model...")
    
    xgb_params = config['model']['xgboost']
    
    model = xgb.XGBRegressor(
        n_estimators=xgb_params['n_estimators'],
        max_depth=xgb_params['max_depth'],
        learning_rate=xgb_params['learning_rate'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        random_state=config['model']['random_state'],
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, 
                               cv=config['training']['cv_folds'], 
                               scoring='neg_mean_squared_error')
    
    logger.info(f"XGBoost CV RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(cv_scores.std() * 2):.2f})")
    
    return model

def train_lightgbm(X_train, y_train, config):
    """Train LightGBM model"""
    logger = logging.getLogger(__name__)
    logger.info("Training LightGBM model...")
    
    lgb_params = config['model']['lightgbm']
    
    model = lgb.LGBMRegressor(
        n_estimators=lgb_params['n_estimators'],
        max_depth=lgb_params['max_depth'],
        learning_rate=lgb_params['learning_rate'],
        num_leaves=lgb_params['num_leaves'],
        feature_fraction=lgb_params['feature_fraction'],
        bagging_fraction=lgb_params['bagging_fraction'],
        random_state=config['model']['random_state'],
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, 
                               cv=config['training']['cv_folds'], 
                               scoring='neg_mean_squared_error')
    
    logger.info(f"LightGBM CV RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(cv_scores.std() * 2):.2f})")
    
    return model

def train_ensemble(X_train, y_train, config):
    """Train ensemble of models"""
    logger = logging.getLogger(__name__)
    logger.info("Training ensemble model...")
    
    # Train individual models
    rf_model = train_random_forest(X_train, y_train, config)
    xgb_model = train_xgboost(X_train, y_train, config)
    lgb_model = train_lightgbm(X_train, y_train, config)
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ])
    
    ensemble.fit(X_train, y_train)
    
    # Cross-validation for ensemble
    cv_scores = cross_val_score(ensemble, X_train, y_train, 
                               cv=config['training']['cv_folds'], 
                               scoring='neg_mean_squared_error')
    
    logger.info(f"Ensemble CV RMSE: {np.sqrt(-cv_scores.mean()):.2f} (+/- {np.sqrt(cv_scores.std() * 2):.2f})")
    
    return ensemble

def get_feature_importance(model, feature_names):
    """Get feature importance from the model"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # For ensemble models, average the importances
        importances = []
        for estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importances.append(estimator.feature_importances_)
        importance = np.mean(importances, axis=0)
    else:
        return None
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def save_model_and_metrics(model, metrics, feature_importance, config):
    """Save trained model and evaluation metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['model']['name']}_{config['model']['type']}_{timestamp}"
    
    # Save model
    model_path = f"models/{model_name}.pkl"
    joblib.dump(model, model_path)
    
    # Save metrics
    os.makedirs('metrics', exist_ok=True)
    metrics_path = f"metrics/{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save feature importance
    if feature_importance is not None:
        importance_path = f"metrics/{model_name}_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
    
    # Save model info
    model_info = {
        'model_name': model_name,
        'model_type': config['model']['type'],
        'timestamp': timestamp,
        'model_path': model_path,
        'metrics_path': metrics_path,
        'performance': metrics
    }
    
    info_path = f"models/{model_name}_info.json"
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return model_path, model_info

def train_model():
    """Main training pipeline"""
    logger = setup_logging()
    logger.info("Starting model training pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    logger.info(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Load feature names
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    # Train model based on configuration
    model_type = config['model']['type']
    
    if model_type == 'random_forest':
        model = train_random_forest(X_train, y_train, config)
    elif model_type == 'xgboost':
        model = train_xgboost(X_train, y_train, config)
    elif model_type == 'lightgbm':
        model = train_lightgbm(X_train, y_train, config)
    elif model_type == 'ensemble':
        model = train_ensemble(X_train, y_train, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    logger.info("Model evaluation results:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    # Get feature importance
    feature_importance = get_feature_importance(model, feature_names)
    if feature_importance is not None:
        logger.info("Top 5 most important features:")
        for idx, row in feature_importance.head().iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # Save model and results
    model_path, model_info = save_model_and_metrics(model, metrics, feature_importance, config)
    
    logger.info(f"Model saved to: {model_path}")
    logger.info("Training pipeline completed successfully!")
    
    return model, metrics, model_info

if __name__ == "__main__":
    train_model()