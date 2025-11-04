import subprocess
import os
import json
import glob
from typing import List, Dict

def get_dvc_tracked_models() -> List[Dict]:
    """
    Get all DVC-tracked model files
    
    Returns:
        List of dictionaries containing model information
    """
    try:
        models = []
        
        # Find all .dvc files for models
        dvc_files = glob.glob('models/*.pkl.dvc')
        
        for dvc_file in dvc_files:
            # Extract model name from .dvc file
            model_pkl_path = dvc_file.replace('.dvc', '')
            model_name = os.path.basename(model_pkl_path).replace('.pkl', '')
            
            # Look for corresponding info file
            info_path = f"models/{model_name}_info.json"
            info_dvc_path = f"{info_path}.dvc"
            
            model_info = {
                'model_name': model_name,
                'model_path': model_pkl_path,
                'dvc_file': dvc_file,
                'info_path': info_path if os.path.exists(info_dvc_path) else None,
                'available_locally': os.path.exists(model_pkl_path),
                'info_available_locally': os.path.exists(info_path),
                'metrics': {}
            }
            
            # Try to read model info if available locally
            if model_info['info_available_locally']:
                try:
                    with open(info_path, 'r') as f:
                        info_data = json.load(f)
                        model_info['model_type'] = info_data.get('model_type', 'Unknown')
                        model_info['timestamp'] = info_data.get('timestamp', 'Unknown')
                        model_info['metrics'] = info_data.get('performance', info_data.get('metrics', {}))
                        model_info['training_samples'] = info_data.get('training_samples', 'Unknown')
                except Exception as e:
                    print(f"Warning: Could not read info for {model_name}: {e}")
            
            models.append(model_info)
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return models
        
    except Exception as e:
        print(f"Error getting DVC tracked models: {e}")
        return []

def pull_model_from_dvc(model_name: str) -> bool:
    """
    Pull a specific model from DVC remote storage
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        model_pkl_path = f"models/{model_name}.pkl"
        model_info_path = f"models/{model_name}_info.json"
        
        # Pull model files from DVC
        pull_commands = [
            ['dvc', 'pull', f"{model_pkl_path}.dvc"],
            ['dvc', 'pull', f"{model_info_path}.dvc"]
        ]
        
        for cmd in pull_commands:
            if os.path.exists(cmd[2]):  # Check if .dvc file exists
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode != 0:
                    print(f"Warning: Could not pull {cmd[2]}: {result.stderr}")
                    return False
                else:
                    print(f"✅ Pulled: {cmd[2]}")
        
        return True
        
    except Exception as e:
        print(f"Error pulling model {model_name}: {e}")
        return False

def switch_to_model(model_name: str) -> Dict:
    """
    Switch to a different DVC-tracked model
    
    Args:
        model_name: Name of the model to switch to
        
    Returns:
        Dict with result information
    """
    try:
        model_pkl_path = f"models/{model_name}.pkl"
        model_info_path = f"models/{model_name}_info.json"
        
        # Check if model exists locally
        if not os.path.exists(model_pkl_path):
            print(f"📥 Model not available locally, pulling from DVC...")
            if not pull_model_from_dvc(model_name):
                return {
                    'success': False,
                    'error': f'Could not pull model {model_name} from DVC remote'
                }
        
        # Verify model files exist
        if not os.path.exists(model_pkl_path):
            return {
                'success': False,
                'error': f'Model file not found: {model_pkl_path}'
            }
        
        # Load model info
        model_info = {}
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
        
        return {
            'success': True,
            'model_name': model_name,
            'model_path': model_pkl_path,
            'info_path': model_info_path,
            'model_info': model_info,
            'message': f'Successfully switched to model: {model_name}'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to switch to model {model_name}: {str(e)}'
        }

def get_model_summary() -> Dict:
    """
    Get a summary of all available models
    
    Returns:
        Dict with model summary information
    """
    try:
        models = get_dvc_tracked_models()
        
        summary = {
            'total_models': len(models),
            'models_available_locally': len([m for m in models if m['available_locally']]),
            'models_in_remote': len(models),
            'latest_model': models[0] if models else None,
            'models': models
        }
        
        return summary
        
    except Exception as e:
        return {
            'error': f'Failed to get model summary: {str(e)}',
            'total_models': 0,
            'models': []
        }