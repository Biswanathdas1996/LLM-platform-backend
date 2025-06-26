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
                    "GET /api/v1/models": "List all models",
                    "POST /api/v1/models": "Upload a new model",
                    "DELETE /api/v1/models/<name>": "Delete a model",
                    "POST /api/v1/models/sync": "Sync models with filesystem"
                },
                "generation": {
                    "POST /api/v1/generate": "Generate text using a model"
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
                    "upload_model": "POST /api/v1/models (with file in form-data)",
                    "list_models": "GET /api/v1/models",
                    "generate_text": "POST /api/v1/generate {\"question\": \"Hello\", \"model_name\": \"model-name\"}"
                }
            },
            "migration_note": "All endpoints are now under /api/v1/. Legacy routes have been removed for cleaner architecture."
        })
    
    return app
