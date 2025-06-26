"""
Model management utilities for handling GGUF models.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Handles model file operations and metadata management."""
    
    def __init__(self, upload_folder: str, models_json_file: str):
        self.upload_folder = upload_folder
        self.models_json_file = models_json_file
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.models_json_file), exist_ok=True)
    
    def load_models_list(self) -> Dict[str, Any]:
        """Load models list from JSON file."""
        if os.path.exists(self.models_json_file):
            try:
                with open(self.models_json_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading models list: {e}")
                return {"models": []}
        return {"models": []}
    
    def save_models_list(self, models_data: Dict[str, Any]) -> bool:
        """Save models list to JSON file."""
        try:
            with open(self.models_json_file, 'w') as f:
                json.dump(models_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving models list: {e}")
            return False
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information including size and modification date."""
        try:
            file_size = os.path.getsize(file_path)
            file_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            return {
                'size': file_size,
                'size_mb': round(file_size / (1024 * 1024), 2),
                'modified_date': file_modified
            }
        except OSError as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {}
    
    def scan_models_folder(self) -> List[Dict[str, Any]]:
        """Scan the models folder and return list of GGUF files."""
        models = []
        if not os.path.exists(self.upload_folder):
            return models
        
        for filename in os.listdir(self.upload_folder):
            if filename.endswith('.gguf'):
                file_path = os.path.join(self.upload_folder, filename)
                file_info = self.get_file_info(file_path)
                
                if file_info:  # Only add if we could get file info
                    model_info = {
                        "filename": filename,
                        "name": filename.replace('.gguf', ''),
                        **file_info
                    }
                    models.append(model_info)
        
        return models
    
    def update_models_list(self) -> Dict[str, Any]:
        """Update models list by scanning the models folder and syncing with JSON."""
        models_data = self.load_models_list()
        current_models = {model['filename']: model for model in models_data.get('models', [])}
        scanned_models = self.scan_models_folder()
        
        updated_models = []
        
        for model_info in scanned_models:
            filename = model_info['filename']
            
            # If model exists in JSON, preserve additional metadata
            if filename in current_models:
                existing_model = current_models[filename]
                # Update with new file info but keep upload_date if it exists
                model_info.update({
                    'upload_date': existing_model.get('upload_date', model_info['modified_date'])
                })
            else:
                # New model found
                model_info['upload_date'] = model_info['modified_date']
            
            updated_models.append(model_info)
        
        # Create updated models data
        models_data = {
            "models": updated_models,
            "last_updated": datetime.now().isoformat(),
            "count": len(updated_models)
        }
        
        self.save_models_list(models_data)
        return models_data
    
    def add_model(self, filename: str, file_path: str) -> Dict[str, Any]:
        """Add a new model to the JSON list."""
        models_data = self.load_models_list()
        file_info = self.get_file_info(file_path)
        current_time = datetime.now().isoformat()
        
        new_model = {
            "filename": filename,
            "name": filename.replace('.gguf', ''),
            "upload_date": current_time,
            **file_info
        }
        
        models_data.setdefault('models', []).append(new_model)
        models_data['last_updated'] = current_time
        models_data['count'] = len(models_data['models'])
        
        self.save_models_list(models_data)
        return new_model
    
    def remove_model(self, filename: str) -> Dict[str, Any]:
        """Remove a model from the JSON list and filesystem."""
        models_data = self.load_models_list()
        file_path = os.path.join(self.upload_folder, filename)
        
        # Remove from filesystem if exists
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted model file: {filename}")
            except OSError as e:
                logger.error(f"Error deleting file {filename}: {e}")
                raise
        
        # Remove from JSON list
        models_data['models'] = [
            m for m in models_data.get('models', []) 
            if m['filename'] != filename
        ]
        models_data['last_updated'] = datetime.now().isoformat()
        models_data['count'] = len(models_data['models'])
        
        self.save_models_list(models_data)
        return models_data
    
    def model_exists(self, filename: str) -> bool:
        """Check if a model file exists."""
        file_path = os.path.join(self.upload_folder, filename)
        return os.path.exists(file_path)
    
    def get_model_path(self, model_name: str) -> str:
        """Get the full path to a model file."""
        if not model_name.endswith('.gguf'):
            model_name += '.gguf'
        return os.path.join(self.upload_folder, model_name)
    
    @staticmethod
    def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
        """Check if file extension is allowed."""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in allowed_extensions
