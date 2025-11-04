from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import plotly
import json
from datetime import datetime, timedelta
import os
import sys

# Add src directory to path to import model selector
sys.path.append('src')
from model_selector import ModelSelector

# Import DVC model management functions
from dvc_model_manager import get_dvc_tracked_models, switch_to_model, get_model_summary

app = Flask(__name__)

# Global variables for model and encoders
model = None
weather_main_encoder = None
weather_description_encoder = None
feature_scaler = None
model_info = None

def load_model_and_encoders():
    """Load the best performing model and encoders using intelligent selection"""
    global model, weather_main_encoder, weather_description_encoder, feature_scaler, model_info
    
    try:
        # Initialize model selector
        selector = ModelSelector()
        
        # Define selection criteria - prioritize R² and RMSE
        selection_criteria = {
            'primary_metric': 'r2',
            'secondary_metric': 'rmse',
            'min_r2': 0.90,  # Minimum R² of 90%
            'max_rmse': 450,  # Maximum RMSE of 450
            'weights': {
                'r2': 0.5,      # 50% weight to R²
                'rmse': 0.3,    # 30% weight to RMSE  
                'mae': 0.2      # 20% weight to MAE
            }
        }
        
        # Select best model
        best_model_info = selector.select_best_model(selection_criteria)
        
        if not best_model_info:
            return None, None, None, None, "No suitable models found!"
        
        # Load model and encoders
        model, weather_main_encoder, weather_description_encoder, feature_scaler = selector.load_model_and_encoders(best_model_info)
        
        if model is None:
            return None, None, None, None, "Failed to load model files!"
        
        model_info = best_model_info
        print(f"✅ Model loaded successfully: {best_model_info.get('model_name', 'Unknown')}")
        return model, weather_main_encoder, weather_description_encoder, feature_scaler, None
        
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"❌ {error_msg}")
        return None, None, None, None, error_msg

def prepare_features(temp, rain, snow, clouds, weather_main, weather_desc, hour, day_of_week, month, is_holiday):
    """Prepare features for prediction"""
    try:
        # Convert temperature from Celsius to original scale
        temp_celsius = float(temp)
        
        # Create weather severity mapping
        weather_severity_map = {
            'Clear': 1, 'Clouds': 2, 'Mist': 3, 'Rain': 4, 'Drizzle': 4,
            'Snow': 5, 'Fog': 3, 'Haze': 3, 'Thunderstorm': 5, 'Smoke': 3
        }
        
        # Encode categorical features
        weather_main_encoded = weather_main_encoder.transform([weather_main])[0] if weather_main_encoder else 0
        weather_desc_encoded = weather_description_encoder.transform([weather_desc])[0] if weather_description_encoder else 0
        
        # Create derived features
        is_weekend = 1 if int(day_of_week) >= 5 else 0
        is_rush_hour = 1 if (7 <= int(hour) <= 9) or (17 <= int(hour) <= 19) else 0
        weather_severity = weather_severity_map.get(weather_main, 2)
        total_precipitation = float(rain) + float(snow)
        
        # Prepare feature array in correct order
        features = np.array([[
            temp_celsius,           # temp_celsius
            float(rain),           # rain_1h
            float(snow),           # snow_1h
            float(clouds),         # clouds_all
            int(hour),             # hour
            int(day_of_week),      # day_of_week
            int(month),            # month
            is_weekend,            # is_weekend
            is_rush_hour,          # is_rush_hour
            int(is_holiday),       # is_holiday
            weather_severity,      # weather_severity
            total_precipitation,   # total_precipitation
            weather_main_encoded,  # weather_main_encoded
            weather_desc_encoded   # weather_description_encoded
        ]])
        
        # Scale features
        if feature_scaler:
            features_scaled = feature_scaler.transform(features)
        else:
            features_scaled = features
            
        return features_scaled
        
    except Exception as e:
        print(f"Error preparing features: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features
        temp = data.get('temperature', 20)
        rain = data.get('rain', 0)
        snow = data.get('snow', 0)
        clouds = data.get('clouds', 50)
        weather_main = data.get('weather_main', 'Clear')
        weather_desc = data.get('weather_description', 'clear sky')
        hour = data.get('hour', 12)
        day_of_week = data.get('day_of_week', 1)
        month = data.get('month', 6)
        is_holiday = data.get('is_holiday', 0)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Prepare features
        features = prepare_features(temp, rain, snow, clouds, weather_main, weather_desc, 
                                  hour, day_of_week, month, is_holiday)
        
        if features is None:
            return jsonify({'error': 'Error preparing features'}), 400
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get confidence interval (simple approach)
        confidence = 0.95 if prediction > 1000 else 0.85
        
        return jsonify({
            'prediction': round(float(prediction)),
            'confidence': confidence,
            'model_name': model_info.get('model_name', 'Unknown') if model_info else 'Unknown'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def get_model_info():
    """Get model information"""
    if model_info:
        return jsonify({
            'model_name': model_info.get('model_name', 'Unknown'),
            'model_type': model_info.get('model_type', 'Unknown'),
            'timestamp': model_info.get('timestamp', 'Unknown'),
            'metrics': model_info.get('metrics', {})
        })
    else:
        return jsonify({'error': 'Model not loaded'}), 500

@app.route('/feature_importance')
def get_feature_importance():
    """Get feature importance visualization"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = [
                'Temperature (°C)', 'Rain (mm)', 'Snow (mm)', 'Cloud Cover (%)',
                'Hour', 'Day of Week', 'Month', 'Is Weekend', 'Is Rush Hour',
                'Is Holiday', 'Weather Severity', 'Total Precipitation',
                'Weather Main', 'Weather Description'
            ]
            
            # Create feature importance plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=importances,
                y=feature_names,
                orientation='h',
                marker=dict(color='skyblue')
            ))
            
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=600
            )
            
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return jsonify({'plot': graphJSON})
        else:
            return jsonify({'error': 'Model does not support feature importance'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain model with new user-provided data using DVC pipeline"""
    global model_info
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'temperature', 'rain', 'snow', 'clouds', 'weather_main', 
            'weather_description', 'hour', 'day_of_week', 'month', 
            'is_holiday', 'actual_traffic_volume'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Import DVC retraining functions
        from dvc_retrain import retrain_with_dvc_pipeline, quick_retrain_fallback
        
        # Try DVC pipeline first
        print("🚀 Attempting DVC-based retraining...")
        dvc_result = retrain_with_dvc_pipeline(data)
        
        if dvc_result['success']:
            # DVC pipeline succeeded
            print("✅ DVC retraining completed successfully!")
            
            # Try to reload the model from the latest DVC output
            try:
                # Reload model and info after DVC pipeline
                latest_model_path = 'models/latest_model.pkl'
                latest_info_path = 'models/latest_model_info.json'
                
                if os.path.exists(latest_model_path) and os.path.exists(latest_info_path):
                    global model
                    model = joblib.load(latest_model_path)
                    
                    with open(latest_info_path, 'r') as f:
                        model_info = json.load(f)
                    
                    print(f"🔄 Reloaded model: {model_info.get('model_name', 'latest_model')}")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not reload model - {str(e)}")
            
            return jsonify({
                'success': True,
                'message': dvc_result['message'],
                'method': 'DVC Pipeline',
                'training_samples': dvc_result.get('training_samples', 'Unknown'),
                'timestamp': dvc_result['timestamp'],
                'metrics': dvc_result.get('metrics', {}),
                'dvc_pipeline': True,
                's3_synced': True,
                'model_info': dvc_result.get('model_info', {})
            })
        
        else:
            # DVC pipeline failed, try fallback method
            print(f"⚠️  DVC retraining failed: {dvc_result.get('error', 'Unknown error')}")
            print("🔄 Falling back to quick retraining method...")
            
            fallback_result = quick_retrain_fallback(data, model, feature_scaler, model_info)
            
            if fallback_result['success']:
                # Update global model info
                model_info = {
                    'model_name': fallback_result.get('model_name', 'quicktrain_model'),
                    'timestamp': fallback_result['timestamp'],
                    'metrics': fallback_result.get('metrics', {})
                }
                
                return jsonify({
                    'success': True,
                    'message': fallback_result['message'],
                    'method': 'Quick Retrain (DVC Fallback)',
                    'training_samples': fallback_result.get('training_samples', 'Unknown'),
                    'timestamp': fallback_result['timestamp'],
                    'metrics': fallback_result.get('metrics', {}),
                    'dvc_pipeline': False,
                    's3_synced': False,
                    'warning': f"DVC pipeline failed: {dvc_result.get('error', 'Unknown error')}"
                })
            else:
                # Both methods failed
                return jsonify({
                    'success': False,
                    'error': 'Both DVC and fallback retraining methods failed',
                    'dvc_error': dvc_result.get('error', 'Unknown DVC error'),
                    'fallback_error': fallback_result.get('error', 'Unknown fallback error')
                }), 500
            
    except Exception as e:
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

@app.route('/models/list')
def list_models():
    """List all DVC-tracked models"""
    try:
        models_summary = get_model_summary()
        return jsonify(models_summary)
    except Exception as e:
        return jsonify({'error': f'Failed to list models: {str(e)}'}), 500

@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different DVC-tracked model"""
    global model, model_info, weather_main_encoder, weather_description_encoder, feature_scaler
    
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400
        
        # Switch to the requested model
        switch_result = switch_to_model(model_name)
        
        if not switch_result['success']:
            return jsonify(switch_result), 500
        
        # Load the new model
        try:
            model = joblib.load(switch_result['model_path'])
            model_info = switch_result['model_info']
            
            # Try to load corresponding encoders (they might not exist for all models)
            try:
                weather_main_encoder = joblib.load('models/weather_main_encoder.pkl')
                weather_description_encoder = joblib.load('models/weather_description_encoder.pkl')  
                feature_scaler = joblib.load('models/feature_scaler.pkl')
            except Exception as e:
                print(f"Warning: Could not load some encoders: {e}")
                # Keep existing encoders if available
            
            print(f"✅ Successfully switched to model: {model_name}")
            
            return jsonify({
                'success': True,
                'message': switch_result['message'],
                'model_name': model_name,
                'model_info': model_info,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to load model {model_name}: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Model switching failed: {str(e)}'}), 500

@app.route('/models/current')
def current_model():
    """Get information about the currently loaded model"""
    try:
        current_info = {
            'model_loaded': model is not None,
            'model_info': model_info if model_info else {},
            'encoders_loaded': {
                'weather_main_encoder': weather_main_encoder is not None,
                'weather_description_encoder': weather_description_encoder is not None,
                'feature_scaler': feature_scaler is not None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(current_info)
        
    except Exception as e:
        return jsonify({'error': f'Failed to get current model info: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Load model on startup
    print("🚀 Starting Traffic Volume Predictor Flask App...")
    load_model_and_encoders()
    
    # Run the app
    app.run(host='0.0.0.0', port=8501, debug=False)