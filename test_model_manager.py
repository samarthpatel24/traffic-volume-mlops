#!/usr/bin/env python3

from dvc_model_manager import get_dvc_tracked_models

def test_model_manager():
    print("🔍 Testing updated DVC model manager...")
    
    try:
        models = get_dvc_tracked_models()
        print(f"\n✅ Found {len(models)} models:")
        
        for i, model in enumerate(models, 1):
            dvc_status = "DVC-tracked" if model["is_dvc_tracked"] else "Local only"
            timestamp = model.get("timestamp", "Unknown")
            model_type = model.get("model_type", "Unknown")
            
            print(f"{i}. {model['model_name']}")
            print(f"   Type: {model_type} | Status: {dvc_status} | Time: {timestamp}")
            print(f"   Available locally: {model['available_locally']}")
            
            if model.get("metrics"):
                print(f"   Metrics: {model['metrics']}")
            print()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_model_manager()