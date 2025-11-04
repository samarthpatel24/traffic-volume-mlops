#!/usr/bin/env python3
"""
Test DVC retraining functionality
"""

import requests
import json

def test_dvc_retrain():
    """Test the DVC-based retraining endpoint"""
    
    # Sample data for retraining
    test_data = {
        'temperature': 15.0,
        'rain': 0.0,
        'snow': 0.0,
        'clouds': 50,
        'weather_main': 'Clear',
        'weather_description': 'clear sky',
        'hour': 14,
        'day_of_week': 1,  # Monday
        'month': 11,  # November
        'is_holiday': False,
        'actual_traffic_volume': 4500
    }
    
    try:
        print("🧪 Testing DVC-based retraining...")
        print(f"📋 Test data: {test_data}")
        
        # Make request to retrain endpoint
        response = requests.post(
            'http://127.0.0.1:8501/retrain',
            json=test_data,
            timeout=300  # 5 minutes timeout for DVC pipeline
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Retraining successful!")
            print(f"📈 Method used: {result.get('method', 'Unknown')}")
            print(f"🔄 DVC Pipeline: {result.get('dvc_pipeline', False)}")
            print(f"☁️  S3 Synced: {result.get('s3_synced', False)}")
            print(f"📊 Training samples: {result.get('training_samples', 'Unknown')}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"📈 Metrics: R²={metrics.get('r2', 'N/A'):.3f}, RMSE={metrics.get('rmse', 'N/A'):.1f}")
                
            if result.get('warning'):
                print(f"⚠️  Warning: {result['warning']}")
                
        else:
            print(f"❌ Retraining failed with status {response.status_code}")
            try:
                error_info = response.json()
                print(f"Error: {error_info}")
            except:
                print(f"Error response: {response.text}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Is it running on http://127.0.0.1:8501?")
    except requests.exceptions.Timeout:
        print("⏰ Request timed out. DVC pipeline might be taking too long.")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_dvc_retrain()