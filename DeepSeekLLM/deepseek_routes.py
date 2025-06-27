"""
DeepSeek API routes.
"""
import logging
from flask import Blueprint, request, jsonify
from .deepseek_service import DeepSeekService

logger = logging.getLogger(__name__)

def create_deepseek_routes(config) -> Blueprint:
    """Create and configure the DeepSeek API blueprint."""
    
    deepseek_bp = Blueprint('deepseek', __name__, url_prefix='/api/deepseek')
    deepseek_service = DeepSeekService(config)
    
    @deepseek_bp.route('/models', methods=['GET'])
    def list_deepseek_models():
        """Get list of all available DeepSeek models."""
        try:
            models = deepseek_service.get_available_models()
            return jsonify({
                "success": True,
                "models": models,
                "count": len(models)
            })
        except Exception as e:
            logger.error(f"Error listing DeepSeek models: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @deepseek_bp.route('/models/<model_name>', methods=['GET'])
    def get_deepseek_model_info(model_name):
        """Get information about a specific DeepSeek model."""
        try:
            model_info = deepseek_service.get_model_info(model_name)
            
            if model_info.get("success"):
                return jsonify(model_info)
            else:
                return jsonify(model_info), 404
                
        except Exception as e:
            logger.error(f"Error getting DeepSeek model info: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @deepseek_bp.route('/generate', methods=['POST'])
    def generate_deepseek_text():
        """Generate text using a DeepSeek model."""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "success": False,
                    "error": "No JSON data provided"
                }), 400
            
            # Required parameters
            model_name = data.get('model_name')
            prompt = data.get('prompt')
            
            if not model_name:
                return jsonify({
                    "success": False,
                    "error": "model_name is required"
                }), 400
                
            if not prompt:
                return jsonify({
                    "success": False,
                    "error": "prompt is required"
                }), 400
            
            # Optional parameters with defaults
            max_tokens = data.get('max_tokens', 200)
            temperature = data.get('temperature', 0.7)
            n_ctx = data.get('n_ctx', 2048)
            n_threads = data.get('n_threads', 8)
            use_mlock = data.get('use_mlock', True)
            stop = data.get('stop')
            
            # Validate parameters
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                return jsonify({
                    "success": False,
                    "error": "max_tokens must be a positive integer"
                }), 400
                
            if not isinstance(temperature, (int, float)) or temperature < 0:
                return jsonify({
                    "success": False,
                    "error": "temperature must be a non-negative number"
                }), 400
            
            # Generate response
            result = deepseek_service.generate_response(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mlock=use_mlock,
                stop=stop
            )
            
            if result.get("success"):
                return jsonify(result)
            else:
                return jsonify(result), 500
                
        except Exception as e:
            logger.error(f"Error generating text with DeepSeek: {e}")
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    @deepseek_bp.route('/health', methods=['GET'])
    def deepseek_health_check():
        """Health check for DeepSeek service."""
        try:
            health_status = deepseek_service.health_check()
            
            if health_status.get("success"):
                return jsonify(health_status)
            else:
                return jsonify(health_status), 503
                
        except Exception as e:
            logger.error(f"Error in DeepSeek health check: {e}")
            return jsonify({
                "success": False,
                "service": "DeepSeek LLM Service",
                "status": "unhealthy",
                "error": str(e)
            }), 503
    
    @deepseek_bp.route('/test', methods=['POST'])
    def test_deepseek_generation():
        """Test endpoint for DeepSeek text generation with default parameters."""
        try:
            data = request.get_json() or {}
            
            # Use default test parameters
            model_name = data.get('model_name', 'DeepSeek-R1-q2_k.gguf')
            prompt = data.get('prompt', 'Hello, how are you?')
            
            result = deepseek_service.generate_response(
                model_name=model_name,
                prompt=prompt,
                max_tokens=50,
                temperature=0.7
            )
            
            return jsonify({
                "test": True,
                "default_model": model_name,
                "default_prompt": prompt,
                **result
            })
            
        except Exception as e:
            logger.error(f"Error in DeepSeek test: {e}")
            return jsonify({
                "success": False,
                "test": True,
                "error": str(e)
            }), 500
    
    return deepseek_bp
