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
from HuggingFace.hf_model_manager import HuggingFaceModelManager
from HuggingFace.hf_service import HuggingFaceService
from HuggingFace.hf_routes import create_hf_routes
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
    
    # Initialize HuggingFace services
    hf_model_manager = HuggingFaceModelManager(
        hf_folder=app_config.HF_FOLDER,
        hf_models_json_file=app_config.HF_MODELS_JSON_FILE
    )
    
    hf_service = HuggingFaceService(app_config, hf_model_manager)
    
    # Register blueprints
    api_blueprint = create_api_blueprint(app_config, model_manager, llm_service)
    app.register_blueprint(api_blueprint)
    
    # Register HuggingFace blueprint
    hf_blueprint = create_hf_routes(hf_model_manager, hf_service)
    app.register_blueprint(hf_blueprint)
    
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
                "huggingface": {
                    "GET /api/huggingface/models": "List all HuggingFace models",
                    "POST /api/huggingface/models": "Add a new HuggingFace model",
                    "GET /api/huggingface/models/<model_id>": "Get specific HuggingFace model",
                    "PUT /api/huggingface/models/<model_id>": "Update HuggingFace model",
                    "DELETE /api/huggingface/models/<model_id>": "Remove HuggingFace model",
                    "POST /api/huggingface/models/<model_id>/load": "Load HuggingFace model into memory",
                    "POST /api/huggingface/generate": "Generate text using HuggingFace model",
                    "POST /api/huggingface/pipeline": "Generate using HuggingFace pipeline",
                    "GET /api/huggingface/cache": "Get cache information",
                    "DELETE /api/huggingface/cache": "Clear model cache",
                    "GET /api/huggingface/statistics": "Get usage statistics",
                    "GET /api/huggingface/model-types": "Get available model types",
                    "GET /api/huggingface/dependencies": "Check required dependencies"
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
                    "add_hf_model": "POST /api/huggingface/models {\"model_id\": \"microsoft/DialoGPT-medium\", \"name\": \"DialoGPT Medium\", \"model_type\": \"conversational\"}",
                    "list_models": "GET /api/v1/models",
                    "list_hf_models": "GET /api/huggingface/models",
                    "generate_text_gguf": "POST /api/v1/generate {\"question\": \"Hello\", \"model_name\": \"model-name\"}",
                    "generate_text_hf": "POST /api/huggingface/generate {\"model_id\": \"microsoft/DialoGPT-medium\", \"prompt\": \"Hello, how are you?\"}"
                }
            },
            "migration_note": "All endpoints are now under /api/v1/ for GGUF models and /api/huggingface/ for HuggingFace models. Legacy routes have been removed for cleaner architecture."
        })
    
    return app
