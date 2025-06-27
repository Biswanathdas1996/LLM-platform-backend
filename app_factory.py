"""
Application factory and initialization.
"""
import os
import logging
from flask import Flask
from flask_cors import CORS
from config import config
from models.model_manager import ModelManager
from services.llm_service import LLMService
from DeepSeekLLM.deepseek_routes import create_deepseek_routes
from api.routes import create_api_blueprint
from utils.logger import APILogger

def setup_logging(app):
    """Set up application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

def create_app(config_name=None):
    """Application factory function."""
    
    # Determine configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    app_config = config[config_name]
    
    # Create Flask app
    app = Flask(__name__)
    app.config.from_object(app_config)
    
    # Set up logging
    setup_logging(app)
    
    # Initialize API logging if enabled
    if app_config.LOG_API_CALLS:
        api_logger = APILogger(app)
    
    # Enable CORS
    CORS(app, origins=app_config.CORS_ORIGINS)
    
    # Initialize services
    model_manager = ModelManager(
        upload_folder=app_config.UPLOAD_FOLDER,
        models_json_file=app_config.MODELS_JSON_FILE
    )
    
    llm_service = LLMService(app_config)
    

    # Register blueprints
    api_blueprint = create_api_blueprint(app_config, model_manager, llm_service)
    app.register_blueprint(api_blueprint)
    
    
    
    # Register DeepSeek blueprint
    deepseek_blueprint = create_deepseek_routes(app_config)
    app.register_blueprint(deepseek_blueprint)
    
    # Root endpoint with API documentation
    @app.route('/', methods=['GET'])
    def api_info():
        """API information and available endpoints."""
        from flask import jsonify
        return jsonify({
            "service": "Local LLM API",
            "version": "2.0.0",
            "status": "running",
            "api_prefix": app_config.API_PREFIX,
            "endpoints": {
                "models": {
                    "GET /api/v1/models": "List all GGUF models",
                    "POST /api/v1/models": "Upload a new GGUF model",
                    "DELETE /api/v1/models/<n>": "Delete a GGUF model",
                    "POST /api/v1/models/sync": "Sync models with filesystem"
                },
                "deepseek": {
                    "GET /api/deepseek/models": "List all DeepSeek models",
                    "GET /api/deepseek/models/<model_name>": "Get specific DeepSeek model info",
                    "POST /api/deepseek/generate": "Generate text using DeepSeek model",
                    "GET /api/deepseek/health": "DeepSeek service health check",
                    "POST /api/deepseek/test": "Test DeepSeek generation with defaults"
                },
                "generation": {
                    "POST /api/v1/generate": "Generate text using a GGUF model"
                },
                "system": {
                    "GET /api/v1/health": "Health check",
                    "POST /api/v1/cache/clear": "Clear model cache",
                    "GET /api/v1/cache/status": "Get cache status"
                }
            },
            "documentation": {
                "base_url": "http://localhost:5000",
                "content_type": "application/json",
                "example_requests": {
                    "upload_gguf_model": "POST /api/v1/models (with file in form-data)",
                    "list_models": "GET /api/v1/models",
                    "list_deepseek_models": "GET /api/deepseek/models",
                    "generate_text_gguf": "POST /api/v1/generate {\"question\": \"Hello\", \"model_name\": \"model-name\"}",
                    "generate_text_deepseek": "POST /api/deepseek/generate {\"model_name\": \"DeepSeek-R1-q2_k.gguf\", \"prompt\": \"Hello, how are you?\", \"max_tokens\": 200, \"temperature\": 0.7}",
                    "test_deepseek": "POST /api/deepseek/test {\"prompt\": \"What is AI?\"}"
                }
            },
        })
    
    return app
