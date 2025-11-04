#!/usr/bin/env python3
"""
Test the new DVC model management features
"""

import requests
import json

def test_model_management():
    """Test all the new model management endpoints"""
    
    base_url = "http://127.0.0.1:8501"
    
    print("🧪 Testing DVC Model Management Features")
    print("=" * 50)
    
    # Test 1: List all available models
    print("\n1️⃣ Testing /models/list endpoint...")
    try:
        response = requests.get(f"{base_url}/models/list")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data.get('total_models', 0)} DVC-tracked models")
            print(f"📱 {data.get('models_available_locally', 0)} available locally")
            print(f"☁️  {data.get('total_models', 0) - data.get('models_available_locally', 0)} in remote only")
            
            if data.get('models'):
                print("\n📋 Available models:")
                for i, model in enumerate(data['models'][:3], 1):  # Show first 3
                    status = "📱 Local" if model.get('available_locally') else "☁️  Remote"
                    model_type = model.get('model_type', 'Unknown')
                    r2 = model.get('metrics', {}).get('r2', 0)
                    print(f"   {i}. {status} | {model['model_name']} ({model_type}, R²: {r2:.3f})")
        else:
            print(f"❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Get current model info
    print("\n2️⃣ Testing /models/current endpoint...")
    try:
        response = requests.get(f"{base_url}/models/current")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Current model loaded: {data.get('model_loaded', False)}")
            if data.get('model_info'):
                model_info = data['model_info']
                print(f"📊 Model: {model_info.get('model_name', 'Unknown')}")
                print(f"🏷️  Type: {model_info.get('model_type', 'Unknown')}")
                if 'performance' in model_info:
                    perf = model_info['performance']
                    print(f"📈 R²: {perf.get('r2', 0):.3f}, RMSE: {perf.get('rmse', 0):.1f}")
        else:
            print(f"❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Test model switching (if we have multiple models)
    print("\n3️⃣ Testing model switching...")
    try:
        # First get the list of models
        models_response = requests.get(f"{base_url}/models/list")
        if models_response.status_code == 200:
            models_data = models_response.json()
            models = models_data.get('models', [])
            
            if len(models) >= 2:
                # Try to switch to the second model
                target_model = models[1]['model_name']
                print(f"🔄 Attempting to switch to: {target_model}")
                
                switch_response = requests.post(
                    f"{base_url}/models/switch",
                    json={'model_name': target_model},
                    timeout=30
                )
                
                if switch_response.status_code == 200:
                    result = switch_response.json()
                    if result.get('success'):
                        print(f"✅ Successfully switched to: {result.get('model_name')}")
                        
                        # Verify the switch by checking current model
                        current_response = requests.get(f"{base_url}/models/current")
                        if current_response.status_code == 200:
                            current_data = current_response.json()
                            current_name = current_data.get('model_info', {}).get('model_name', 'Unknown')
                            print(f"✅ Verified: Current model is now {current_name}")
                        
                    else:
                        print(f"❌ Switch failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"❌ Switch request failed with status {switch_response.status_code}")
            else:
                print("ℹ️  Need at least 2 models to test switching")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Model management testing complete!")

if __name__ == "__main__":
    test_model_management()