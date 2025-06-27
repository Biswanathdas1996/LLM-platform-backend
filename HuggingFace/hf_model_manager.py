"""
Hugging Face model manager for handling HF model operations and metadata.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class HuggingFaceModelManager:
    """Handles Hugging Face model operations and metadata management."""
    
    def __init__(self, hf_folder: str, hf_models_json_file: str):
        self.hf_folder = hf_folder
        self.hf_models_json_file = hf_models_json_file
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        os.makedirs(self.hf_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.hf_models_json_file), exist_ok=True)
    
    def load_hf_models_list(self) -> Dict[str, Any]:
        """Load Hugging Face models list from JSON file."""
        if os.path.exists(self.hf_models_json_file):
            try:
                with open(self.hf_models_json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error loading HF models list: {e}")
                return {"models": []}
        return {"models": []}
    
    def save_hf_models_list(self, models_data: Dict[str, Any]) -> bool:
        """Save Hugging Face models list to JSON file."""
        try:
            with open(self.hf_models_json_file, 'w', encoding='utf-8') as f:
                json.dump(models_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving HF models list: {e}")
            return False
    
    def add_hf_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new Hugging Face model to the JSON list.
        
        Args:
            model_info: Dictionary containing model information with required fields:
                - model_id: Hugging Face model identifier (e.g., "microsoft/DialoGPT-medium")
                - name: Display name for the model
                - model_type: Type of model (e.g., "text-generation", "conversational")
                - description: Optional description
                - parameters: Optional model parameters (temperature, max_tokens, etc.)
        """
        models_data = self.load_hf_models_list()
        current_time = datetime.now().isoformat()
        
        # Validate required fields
        required_fields = ['model_id', 'name', 'model_type']
        for field in required_fields:
            if field not in model_info:
                raise ValueError(f"Missing required field: {field}")
        
        # Check if model already exists
        existing_models = models_data.get('models', [])
        for existing_model in existing_models:
            if existing_model['model_id'] == model_info['model_id']:
                raise ValueError(f"Model with ID '{model_info['model_id']}' already exists")
        
        new_model = {
            "model_id": model_info['model_id'],
            "name": model_info['name'],
            "model_type": model_info['model_type'],
            "description": model_info.get('description', ''),
            "parameters": model_info.get('parameters', {}),
            "added_date": current_time,
            "last_used": None,
            "usage_count": 0,
            "status": "available"
        }
        
        models_data.setdefault('models', []).append(new_model)
        models_data['last_updated'] = current_time
        models_data['count'] = len(models_data['models'])
        
        self.save_hf_models_list(models_data)
        logger.info(f"Added HF model: {model_info['model_id']}")
        return new_model
    
    def remove_hf_model(self, model_id: str) -> Dict[str, Any]:
        """Remove a Hugging Face model from the JSON list."""
        models_data = self.load_hf_models_list()
        
        # Find and remove the model
        original_count = len(models_data.get('models', []))
        models_data['models'] = [
            m for m in models_data.get('models', []) 
            if m['model_id'] != model_id
        ]
        
        if len(models_data['models']) == original_count:
            raise ValueError(f"Model with ID '{model_id}' not found")
        
        models_data['last_updated'] = datetime.now().isoformat()
        models_data['count'] = len(models_data['models'])
        
        self.save_hf_models_list(models_data)
        logger.info(f"Removed HF model: {model_id}")
        return models_data
    
    def get_hf_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific Hugging Face model by ID."""
        models_data = self.load_hf_models_list()
        for model in models_data.get('models', []):
            if model['model_id'] == model_id:
                return model
        return None
    
    def update_model_usage(self, model_id: str) -> bool:
        """Update usage statistics for a model."""
        models_data = self.load_hf_models_list()
        current_time = datetime.now().isoformat()
        
        for model in models_data.get('models', []):
            if model['model_id'] == model_id:
                model['last_used'] = current_time
                model['usage_count'] = model.get('usage_count', 0) + 1
                models_data['last_updated'] = current_time
                self.save_hf_models_list(models_data)
                return True
        return False
    
    def list_hf_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all Hugging Face models, optionally filtered by type.
        
        Args:
            model_type: Optional filter by model type
        """
        models_data = self.load_hf_models_list()
        models = models_data.get('models', [])
        
        if model_type:
            models = [m for m in models if m.get('model_type') == model_type]
        
        return models
    
    def update_model_info(self, model_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update model information."""
        models_data = self.load_hf_models_list()
        
        for model in models_data.get('models', []):
            if model['model_id'] == model_id:
                # Prevent updating certain protected fields
                protected_fields = ['model_id', 'added_date', 'usage_count']
                for key, value in updates.items():
                    if key not in protected_fields:
                        model[key] = value
                
                models_data['last_updated'] = datetime.now().isoformat()
                self.save_hf_models_list(models_data)
                return model
        
        raise ValueError(f"Model with ID '{model_id}' not found")
    
    def get_model_types(self) -> List[str]:
        """Get list of unique model types."""
        models_data = self.load_hf_models_list()
        model_types = set()
        for model in models_data.get('models', []):
            model_types.add(model.get('model_type', 'unknown'))
        return sorted(list(model_types))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the Hugging Face models."""
        models_data = self.load_hf_models_list()
        models = models_data.get('models', [])
        
        total_usage = sum(model.get('usage_count', 0) for model in models)
        model_types = {}
        
        for model in models:
            model_type = model.get('model_type', 'unknown')
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        return {
            'total_models': len(models),
            'total_usage': total_usage,
            'model_types': model_types,
            'last_updated': models_data.get('last_updated'),
            'most_used': max(models, key=lambda x: x.get('usage_count', 0)) if models else None
        }
