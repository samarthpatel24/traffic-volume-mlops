import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelSelector:
    """Intelligent model selection utility for the Traffic Volume Predictor"""
    
    def __init__(self, models_dir='models', metrics_dir='metrics'):
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        
    def get_available_models(self):
        """Get all available trained models"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
            
        for file in os.listdir(self.models_dir):
            if file.endswith('_info.json'):
                try:
                    with open(os.path.join(self.models_dir, file), 'r') as f:
                        model_info = json.load(f)
                    models.append(model_info)
                except Exception as e:
                    print(f"Error loading model info from {file}: {e}")
                    
        return models
    
    def load_model_metrics(self, model_name):
        """Load metrics for a specific model"""
        metrics_file = os.path.join(self.metrics_dir, f"{model_name}_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return None
    
    def calculate_model_score(self, metrics, weights=None):
        """Calculate a composite score for model ranking"""
        if weights is None:
            weights = {'r2': 0.5, 'rmse': -0.3, 'mae': -0.2}
        
        score = 0
        if 'r2' in metrics and 'r2' in weights:
            score += metrics['r2'] * weights['r2']
        if 'rmse' in metrics and 'rmse' in weights:
            # Normalize RMSE (lower is better, so negative weight)
            normalized_rmse = 1000 / (metrics['rmse'] + 1)  # +1 to avoid division by zero
            score += normalized_rmse * abs(weights['rmse'])
        if 'mae' in metrics and 'mae' in weights:
            # Normalize MAE (lower is better, so negative weight) 
            normalized_mae = 1000 / (metrics['mae'] + 1)
            score += normalized_mae * abs(weights['mae'])
            
        return score
    
    def select_best_model(self, selection_criteria=None, fallback_to_latest=True):
        """Select the best model based on criteria"""
        models = self.get_available_models()
        
        if not models:
            if fallback_to_latest and self._has_latest_model():
                return self._get_latest_model_info()
            return None
        
        # Default selection criteria
        if selection_criteria is None:
            selection_criteria = {
                'primary_metric': 'r2',
                'secondary_metric': 'rmse',
                'weights': {'r2': 0.5, 'rmse': -0.3, 'mae': -0.2}
            }
        
        best_model = None
        best_score = -float('inf')
        
        for model in models:
            model_name = model.get('model_name')
            if not model_name:
                continue
                
            # Load model metrics
            metrics = self.load_model_metrics(model_name)
            if not metrics:
                # Try to get metrics from model_info performance field
                metrics = model.get('performance', {})
            
            if not metrics:
                continue
            
            # Check minimum requirements
            if 'min_r2' in selection_criteria and metrics.get('r2', 0) < selection_criteria['min_r2']:
                continue
            if 'max_rmse' in selection_criteria and metrics.get('rmse', float('inf')) > selection_criteria['max_rmse']:
                continue
            
            # Calculate composite score
            score = self.calculate_model_score(metrics, selection_criteria.get('weights'))
            
            if score > best_score:
                best_score = score
                best_model = model
                best_model['metrics'] = metrics
                best_model['score'] = score
        
        # Fallback to latest model if no model meets criteria
        if best_model is None and fallback_to_latest:
            return self._get_latest_model_info()
            
        return best_model
    
    def _has_latest_model(self):
        """Check if latest model files exist"""
        latest_model_path = os.path.join(self.models_dir, 'latest_model.pkl')
        latest_info_path = os.path.join(self.models_dir, 'latest_model_info.json')
        return os.path.exists(latest_model_path) and os.path.exists(latest_info_path)
    
    def _get_latest_model_info(self):
        """Get latest model info as fallback"""
        if not self._has_latest_model():
            return None
            
        try:
            with open(os.path.join(self.models_dir, 'latest_model_info.json'), 'r') as f:
                model_info = json.load(f)
            
            # Try to load latest metrics
            latest_metrics_path = os.path.join(self.metrics_dir, 'latest_metrics.json')
            if os.path.exists(latest_metrics_path):
                with open(latest_metrics_path, 'r') as f:
                    metrics = json.load(f)
                model_info['metrics'] = metrics
            
            return model_info
        except Exception as e:
            print(f"Error loading latest model info: {e}")
            return None
    
    def compare_models(self):
        """Compare all available models and return a DataFrame"""
        models = self.get_available_models()
        
        if not models:
            return pd.DataFrame()
        
        comparison_data = []
        for model in models:
            model_name = model.get('model_name', 'Unknown')
            model_type = model.get('model_type', 'Unknown')
            timestamp = model.get('timestamp', 'Unknown')
            
            # Load metrics
            metrics = self.load_model_metrics(model_name)
            if not metrics:
                metrics = model.get('performance', {})
            
            row = {
                'Model Name': model_name,
                'Model Type': model_type,
                'Timestamp': timestamp,
                'R² Score': metrics.get('r2', 'N/A'),
                'RMSE': metrics.get('rmse', 'N/A'),
                'MAE': metrics.get('mae', 'N/A'),
                'MAPE': metrics.get('mape', 'N/A'),
                'Score': self.calculate_model_score(metrics) if metrics else 0
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        if not df.empty and 'Score' in df.columns:
            df = df.sort_values('Score', ascending=False)
        
        return df
    
    def get_model_selection_config(self):
        """Get recommended model selection configuration"""
        return {
            'primary_metric': 'r2',
            'secondary_metric': 'rmse', 
            'min_r2': 0.90,
            'max_rmse': 450,
            'weights': {
                'r2': 0.5,
                'rmse': -0.3,
                'mae': -0.2
            },
            'description': {
                'r2': 'Coefficient of determination (higher is better)',
                'rmse': 'Root Mean Square Error (lower is better)',
                'mae': 'Mean Absolute Error (lower is better)',
                'mape': 'Mean Absolute Percentage Error (lower is better)'
            }
        }
    
    def load_model_and_encoders(self, model_info=None):
        """Load model and associated encoders"""
        if model_info is None:
            model_info = self.select_best_model()
        
        if not model_info:
            return None, None, None, None
        
        try:
            # Load model
            model_path = model_info.get('model_path')
            if not model_path or not os.path.exists(model_path):
                # Try latest model as fallback
                model_path = os.path.join(self.models_dir, 'latest_model.pkl')
                if not os.path.exists(model_path):
                    return None, None, None, None
            
            model = joblib.load(model_path)
            
            # Load encoders
            encoders = {}
            encoder_files = {
                'weather_main': os.path.join(self.models_dir, 'weather_main_encoder.pkl'),
                'weather_description': os.path.join(self.models_dir, 'weather_description_encoder.pkl'),
                'scaler': os.path.join(self.models_dir, 'feature_scaler.pkl')
            }
            
            for encoder_name, encoder_path in encoder_files.items():
                if os.path.exists(encoder_path):
                    encoders[encoder_name] = joblib.load(encoder_path)
                else:
                    print(f"Warning: {encoder_name} not found at {encoder_path}")
            
            return model, encoders.get('weather_main'), encoders.get('weather_description'), encoders.get('scaler')
            
        except Exception as e:
            print(f"Error loading model and encoders: {e}")
            return None, None, None, None