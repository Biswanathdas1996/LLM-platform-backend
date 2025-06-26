"""
API routes for the Local LLM application.
"""
import logging
from flask import Blueprint
from services.route_handlers import RouteHandlers

logger = logging.getLogger(__name__)

def create_api_blueprint(config, model_manager, llm_service) -> Blueprint:
    """Create and configure the API blueprint."""
    
    api = Blueprint('api', __name__, url_prefix=config.API_PREFIX)
    
    # Create shared route handlers
    handlers = RouteHandlers(model_manager, llm_service, config)
    
    @api.route('/models', methods=['GET'])
    def list_models():
        """Get list of all available models."""
        return handlers.list_models()
    
    @api.route('/models', methods=['POST'])
    def upload_model():
        """Upload a new model file."""
        return handlers.upload_model()
    
    @api.route('/models/<model_name>', methods=['DELETE'])
    def delete_model(model_name):
        """Delete a model."""
        return handlers.delete_model(model_name)
    
    @api.route('/models/sync', methods=['POST'])
    def sync_models():
        """Manually sync models list with filesystem."""
        return handlers.sync_models()
    
    @api.route('/generate', methods=['POST'])
    def generate_response():
        """Generate a response using the specified model."""
        return handlers.generate_response()
    
    @api.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return handlers.health_check()
    
    @api.route('/cache/clear', methods=['POST'])
    def clear_cache():
        """Clear the LLM cache."""
        return handlers.clear_cache()
    
    @api.route('/cache/status', methods=['GET'])
    def cache_status():
        """Get cache status."""
        return handlers.cache_status()
    
    @api.route('/upload', methods=['POST'])
    def upload_model_api():
        """Get cache status."""
        return handlers.upload_model()
    
    @api.route('/logs', methods=['GET'])
    def get_logs():
        """Get recent API and error logs."""
        return handlers.get_logs()
    
    @api.route('/clear-logs', methods=['GET'])
    def clear_logs():
        """Get recent API and error logs."""
        return handlers.clear_logs()
    
    return api
