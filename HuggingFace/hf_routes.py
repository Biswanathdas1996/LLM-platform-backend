"""
API routes for Hugging Face model management.
"""
from flask import Blueprint, request, jsonify
import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_hf_routes(hf_model_manager, hf_service) -> Blueprint:
    """Create Hugging Face API routes."""
    
    hf_bp = Blueprint('huggingface', __name__, url_prefix='/api/huggingface')
    
    @hf_bp.route('/models', methods=['GET'])
    def list_hf_models():
        """List all Hugging Face models."""
        try:
            model_type = request.args.get('type')
            models = hf_model_manager.list_hf_models(model_type)
            return jsonify({
                "success": True,
                "models": models,
                "count": len(models)
            })
        except Exception as e:
            logger.error(f"Error listing HF models: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models', methods=['POST'])
    def add_hf_model():
        """Add a new Hugging Face model."""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No JSON data provided"
                }), 400
            
            # Validate required fields
            required_fields = ['model_id', 'name', 'model_type']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }), 400
            
            model = hf_model_manager.add_hf_model(data)
            return jsonify({
                "success": True,
                "model": model,
                "message": f"Model '{data['model_id']}' added successfully"
            })
            
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
        except Exception as e:
            logger.error(f"Error adding HF model: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models/<path:model_id>', methods=['GET'])
    def get_hf_model(model_id: str):
        """Get a specific Hugging Face model."""
        try:
            model = hf_model_manager.get_hf_model(model_id)
            if model:
                return jsonify({
                    "success": True,
                    "model": model
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"Model '{model_id}' not found"
                }), 404
        except Exception as e:
            logger.error(f"Error getting HF model: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models/<path:model_id>', methods=['DELETE'])
    def remove_hf_model(model_id: str):
        """Remove a Hugging Face model."""
        try:
            result = hf_model_manager.remove_hf_model(model_id)
            return jsonify({
                "success": True,
                "message": f"Model '{model_id}' removed successfully",
                "models_data": result
            })
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 404
        except Exception as e:
            logger.error(f"Error removing HF model: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models/<path:model_id>', methods=['PUT'])
    def update_hf_model(model_id: str):
        """Update a Hugging Face model."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No JSON data provided"
                }), 400
            
            model = hf_model_manager.update_model_info(model_id, data)
            return jsonify({
                "success": True,
                "model": model,
                "message": f"Model '{model_id}' updated successfully"
            })
        except ValueError as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 404
        except Exception as e:
            logger.error(f"Error updating HF model: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models/<path:model_id>/load', methods=['POST'])
    async def load_hf_model(model_id: str):
        """Load a Hugging Face model into memory."""
        try:
            data = request.get_json() or {}
            force_reload = data.get('force_reload', False)
            
            result = await hf_service.load_model(model_id, force_reload)
            
            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400
        except Exception as e:
            logger.error(f"Error loading HF model: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/generate', methods=['POST'])
    async def generate_hf_response():
        """Generate response using a Hugging Face model."""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No JSON data provided"
                }), 400
            
            model_id = data.get('model_id')
            prompt = data.get('prompt')
            
            if not model_id or not prompt:
                return jsonify({
                    "success": False,
                    "error": "Both 'model_id' and 'prompt' are required"
                }), 400
            
            # Extract generation parameters
            generation_params = {
                k: v for k, v in data.items() 
                if k not in ['model_id', 'prompt']
            }
            
            result = await hf_service.generate_response(
                model_id, prompt, **generation_params
            )
            
            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"Error generating HF response: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/pipeline', methods=['POST'])
    async def generate_hf_pipeline():
        """Generate response using Hugging Face pipeline."""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No JSON data provided"
                }), 400
            
            model_id = data.get('model_id')
            task = data.get('task')
            inputs = data.get('inputs')
            
            if not all([model_id, task, inputs]):
                return jsonify({
                    "success": False,
                    "error": "'model_id', 'task', and 'inputs' are required"
                }), 400
            
            # Extract pipeline parameters
            pipeline_params = {
                k: v for k, v in data.items() 
                if k not in ['model_id', 'task', 'inputs']
            }
            
            result = await hf_service.generate_with_pipeline(
                model_id, task, inputs, **pipeline_params
            )
            
            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"Error with HF pipeline: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/cache', methods=['GET'])
    def get_hf_cache_info():
        """Get information about cached models."""
        try:
            cache_info = hf_service.get_cache_info()
            return jsonify({
                "success": True,
                **cache_info
            })
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/cache', methods=['DELETE'])
    def clear_hf_cache():
        """Clear model cache."""
        try:
            model_id = request.args.get('model_id')
            result = hf_service.clear_cache(model_id)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/statistics', methods=['GET'])
    def get_hf_statistics():
        """Get Hugging Face models statistics."""
        try:
            stats = hf_model_manager.get_statistics()
            return jsonify({
                "success": True,
                "statistics": stats
            })
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/model-types', methods=['GET'])
    def get_hf_model_types():
        """Get list of unique model types."""
        try:
            model_types = hf_model_manager.get_model_types()
            return jsonify({
                "success": True,
                "model_types": model_types
            })
        except Exception as e:
            logger.error(f"Error getting model types: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/dependencies', methods=['GET'])
    def check_hf_dependencies():
        """Check if required dependencies are available."""
        try:
            deps = hf_service.check_dependencies()
            return jsonify({
                "success": True,
                "dependencies": deps
            })
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models/<path:model_id>/gguf', methods=['GET'])
    def list_gguf_files(model_id: str):
        """List available GGUF files for a model."""
        try:
            result = asyncio.run(hf_service.list_gguf_files(model_id))
            
            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"Error listing GGUF files for model '{model_id}': {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @hf_bp.route('/models/<path:model_id>/gguf/download', methods=['POST'])
    def download_gguf_model(model_id: str):
        """Download a GGUF model from HuggingFace Hub."""
        try:
            data = request.get_json() or {}
            filename = data.get('filename')
            download_path = data.get('download_path')
            
            result = asyncio.run(hf_service.download_gguf_model(
                model_id=model_id,
                filename=filename,
                download_path=download_path
            ))
            
            if result["success"]:
                return jsonify(result)
            else:
                return jsonify(result), 400
                
        except Exception as e:
            logger.error(f"Error downloading GGUF model '{model_id}': {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500

    return hf_bp
