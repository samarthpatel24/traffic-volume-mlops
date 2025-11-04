import subprocess
import os
import json
import pandas as pd
from datetime import datetime
import tempfile

def retrain_with_dvc_pipeline(new_data_point):
    """
    Retrain model using DVC pipeline with new data point
    
    Args:
        new_data_point (dict): Dictionary containing new data fields
        
    Returns:
        dict: Result of retraining operation
    """
    try:
        print("🔄 Starting DVC-based retraining...")
        
        # 1. Add new data to the raw dataset
        raw_data_path = 'data/raw/Metro_Interstate_Traffic_Volume.csv'
        if not os.path.exists(raw_data_path):
            return {'success': False, 'error': 'Raw data file not found'}
        
        # Read existing raw data
        raw_data = pd.read_csv(raw_data_path)
        
        # Create new row for raw data (convert from processed format back to raw format)
        new_raw_row = {
            'holiday': 'None' if not new_data_point.get('is_holiday', False) else 'Labor Day',  # Simplified
            'temp': new_data_point['temperature'] + 273.15,  # Convert Celsius back to Kelvin
            'rain_1h': new_data_point['rain'],
            'snow_1h': new_data_point['snow'],
            'clouds_all': new_data_point['clouds'],
            'weather_main': new_data_point['weather_main'],
            'weather_description': new_data_point['weather_description'],
            'date_time': datetime.now().strftime('%d-%m-%Y %H:%M'),  # Correct format: DD-MM-YYYY HH:MM
            'traffic_volume': new_data_point['actual_traffic_volume']
        }
        
        # Add new row to raw data
        new_raw_data = pd.concat([raw_data, pd.DataFrame([new_raw_row])], ignore_index=True)
        
        # Save updated raw data
        backup_path = f"{raw_data_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        raw_data.to_csv(backup_path, index=False)  # Create backup
        new_raw_data.to_csv(raw_data_path, index=False)
        
        print(f"✅ Added new data point to raw dataset ({len(new_raw_data)} total rows)")
        
        # 2. Run DVC pipeline to retrain
        print("🚀 Running DVC pipeline...")
        
        # Run the complete DVC pipeline
        dvc_commands = [
            ['dvc', 'repro', '--force'],  # Force reproduction of all stages
            ['dvc', 'push']  # Push results to S3
        ]
        
        pipeline_success = True
        for cmd in dvc_commands:
            print(f"📋 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                print(f"❌ Command failed: {' '.join(cmd)}")
                print(f"Error: {result.stderr}")
                pipeline_success = False
                break
            else:
                print(f"✅ Command completed: {' '.join(cmd)}")
                if result.stdout.strip():
                    print(f"Output: {result.stdout.strip()}")
        
        if not pipeline_success:
            # Restore backup on failure
            if os.path.exists(backup_path):
                pd.read_csv(backup_path).to_csv(raw_data_path, index=False)
                print("🔄 Restored backup due to pipeline failure")
            return {'success': False, 'error': 'DVC pipeline execution failed'}
        
        # 3. Load and return information about the new model
        try:
            # Read latest model info
            latest_model_info_path = 'models/latest_model_info.json'
            if os.path.exists(latest_model_info_path):
                with open(latest_model_info_path, 'r') as f:
                    model_info = json.load(f)
            else:
                model_info = {'model_name': 'latest_model', 'metrics': {}}
            
            # Read latest metrics
            latest_metrics_path = 'metrics/latest_metrics.json'
            if os.path.exists(latest_metrics_path):
                with open(latest_metrics_path, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {}
            
            print("✅ DVC pipeline completed successfully!")
            print(f"📊 New model metrics: {metrics}")
            
            return {
                'success': True,
                'message': 'Model retrained successfully using DVC pipeline',
                'model_info': model_info,
                'metrics': metrics,
                'training_samples': len(new_raw_data),
                'timestamp': datetime.now().isoformat(),
                'dvc_pipeline': True,
                's3_synced': True
            }
            
        except Exception as e:
            print(f"⚠️  Warning: Could not read model info - {str(e)}")
            return {
                'success': True,
                'message': 'Model retrained successfully using DVC pipeline (info read failed)',
                'training_samples': len(new_raw_data),
                'timestamp': datetime.now().isoformat(),
                'dvc_pipeline': True,
                's3_synced': True,
                'warning': f'Could not read model info: {str(e)}'
            }
            
    except Exception as e:
        print(f"❌ DVC retraining failed: {str(e)}")
        return {'success': False, 'error': f'DVC retraining failed: {str(e)}'}

def quick_retrain_fallback(new_data_point, model, feature_scaler, model_info):
    """
    Fallback quick retraining method (existing implementation) if DVC fails
    """
    try:
        print("🔄 Using fallback quick retraining...")
        
        # Prepare features for the new data point
        from flask_app import prepare_features
        
        features = prepare_features(
            new_data_point['temperature'], new_data_point['rain'], 
            new_data_point['snow'], new_data_point['clouds'],
            new_data_point['weather_main'], new_data_point['weather_description'], 
            new_data_point['hour'], new_data_point['day_of_week'], 
            new_data_point['month'], new_data_point['is_holiday']
        )
        
        if features is None:
            return {'success': False, 'error': 'Failed to prepare features'}
        
        # Load existing training data
        train_data_path = 'data/processed/train.csv'
        if not os.path.exists(train_data_path):
            return {'success': False, 'error': 'Training data not found'}
        
        existing_data = pd.read_csv(train_data_path)
        
        # Create new data point as DataFrame
        feature_names = [
            'temp_celsius', 'rain_1h', 'snow_1h', 'clouds_all',
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            'is_holiday', 'weather_severity', 'total_precipitation',
            'weather_main_encoded', 'weather_description_encoded'
        ]
        
        new_row = {}
        for i, name in enumerate(feature_names):
            new_row[name] = features[0][i]
        new_row['traffic_volume'] = float(new_data_point['actual_traffic_volume'])
        
        # Add new data point
        new_data_df = pd.DataFrame([new_row])
        updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)
        
        # Save updated training data
        updated_data.to_csv(train_data_path, index=False)
        
        # Retrain model
        X = updated_data[feature_names]
        y = updated_data['traffic_volume']
        
        if feature_scaler:
            X_scaled = feature_scaler.fit_transform(X)
        else:
            X_scaled = X
        
        model.fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = model.predict(X_scaled)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import math
        
        mse = mean_squared_error(y, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Save model
        import joblib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"traffic_volume_predictor_quicktrain_{timestamp}"
        model_path = f"models/{model_name}.pkl"
        
        joblib.dump(model, model_path)
        
        print(f"✅ Quick retraining completed: {model_name}")
        
        return {
            'success': True,
            'message': 'Model retrained successfully (quick method)',
            'model_name': model_name,
            'training_samples': len(updated_data),
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'r2': float(r2),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'dvc_pipeline': False,
            's3_synced': False
        }
        
    except Exception as e:
        return {'success': False, 'error': f'Quick retraining failed: {str(e)}'}