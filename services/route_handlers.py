"""
Route handlers for both API and legacy endpoints.
This module contains the business logic that can be shared between different route implementations.
"""
import asyncio
import logging
from flask import request, jsonify
from werkzeug.utils import secure_filename
from utils.logger import log_endpoint, log_custom_event

logger = logging.getLogger(__name__)

class RouteHandlers:
    """Shared route handlers to eliminate code duplication."""
    
    def __init__(self, model_manager, llm_service, app_config):
        self.model_manager = model_manager
        self.llm_service = llm_service
        self.app_config = app_config
    
    @log_endpoint("upload_model")
    def upload_model(self):
        """Handle model upload requests."""
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file or not self.model_manager.is_allowed_file(file.filename, self.app_config.ALLOWED_EXTENSIONS):
            return jsonify({
                "error": f"Invalid file type. Only {', '.join(self.app_config.ALLOWED_EXTENSIONS)} files are allowed"
            }), 400
        
        filename = secure_filename(file.filename)
        
        # Check if file already exists
        if self.model_manager.model_exists(filename):
            return jsonify({"error": "File already exists"}), 409
        
        try:
            file_path = self.model_manager.get_model_path(filename)
            file.save(file_path)
            model_info = self.model_manager.add_model(filename, file_path)
            
            # Log successful upload
            log_custom_event(
                "model_upload", 
                f"Model uploaded successfully: {filename}",
                {
                    "filename": filename,
                    "file_size": file.content_length,
                    "model_info": model_info
                }
            )
            
            logger.info(f"Model uploaded successfully: {filename}")
            return jsonify({
                "message": "File uploaded successfully",
                "filename": filename,
                "model_info": model_info
            }), 200
            
        except Exception as e:
            # Log the error with context
            log_custom_event(
                "model_upload_error",
                f"Failed to upload model: {str(e)}",
                {
                    "filename": file.filename if 'file' in locals() else 'unknown',
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            logger.error(f"Error uploading model: {e}")
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    
    @log_endpoint("list_models")
    def list_models(self):
        """Handle list models requests."""
        try:
            models_data = self.model_manager.update_models_list()
            
            # Log successful model listing
            log_custom_event(
                "models_listed",
                f"Successfully listed {models_data.get('count', 0)} models",
                {"model_count": models_data.get('count', 0)}
            )
            
            return jsonify(models_data), 200
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return jsonify({"error": f"Failed to list models: {str(e)}"}), 500
    
    @log_endpoint("delete_model")
    def delete_model(self, model_name):
        """Handle delete model requests."""
        try:
            # Add .gguf extension if not present
            if not model_name.endswith('.gguf'):
                filename = f"{model_name}.gguf"
            else:
                filename = model_name
            
            # Check if file exists
            if not self.model_manager.model_exists(filename):
                return jsonify({"error": "Model not found"}), 404
            
            # Delete the model
            updated_data = self.model_manager.remove_model(filename)
            
            # Log successful deletion
            log_custom_event(
                "model_deleted",
                f"Model deleted successfully: {filename}",
                {
                    "filename": filename,
                    "remaining_models": updated_data['count']
                }
            )
            
            logger.info(f"Model deleted successfully: {filename}")
            return jsonify({
                "message": "Model deleted successfully",
                "filename": filename,
                "remaining_models": updated_data['count']
            }), 200
            
        except Exception as e:
            log_custom_event(
                "model_deletion_error",
                f"Failed to delete model {model_name}: {str(e)}",
                {
                    "model_name": model_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            logger.error(f"Error deleting model: {e}")
            return jsonify({"error": f"Failed to delete model: {str(e)}"}), 500
    
    @log_endpoint("sync_models")
    def sync_models(self):
        """Handle sync models requests."""
        try:
            models_data = self.model_manager.update_models_list()
            
            # Log successful sync
            log_custom_event(
                "models_synced",
                f"Models synchronized successfully: {models_data.get('count', 0)} models",
                {"model_count": models_data.get('count', 0)}
            )
            
            return jsonify({
                "message": "Models list synchronized successfully",
                "data": models_data
            }), 200
        except Exception as e:
            logger.error(f"Error syncing models: {e}")
            return jsonify({"error": f"Failed to sync models: {str(e)}"}), 500
    
    @log_endpoint("generate_response")
    def generate_response(self):
        """Handle text generation requests."""
        try:
            data = request.get_json()
            
            # Validate request data
            if not data:
                return jsonify({"error": "Request body is required"}), 400
            
            question = data.get('question')
            model_name = data.get('model_name')
            
            if not question:
                return jsonify({"error": "Question is required"}), 400
            
            if not model_name:
                return jsonify({"error": "Model name is required"}), 400
            
            # Get model path
            model_path = self.model_manager.get_model_path(model_name)
            
            # Check if model exists
            if not self.model_manager.model_exists(model_name + '.gguf' if not model_name.endswith('.gguf') else model_name):
                return jsonify({"error": "Model not found"}), 404
            
            # Extract additional parameters
            template = data.get('template')
            llm_params = {
                'n_gpu_layers': data.get('n_gpu_layers', self.app_config.DEFAULT_N_GPU_LAYERS),
                'n_batch': data.get('n_batch', self.app_config.DEFAULT_N_BATCH),
                'temperature': data.get('temperature')
            }
            
            # Remove None values
            llm_params = {k: v for k, v in llm_params.items() if v is not None}
            
            # Run the async function in an event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Log generation request
                log_custom_event(
                    "text_generation_start",
                    f"Starting text generation with model: {model_name}",
                    {
                        "model_name": model_name,
                        "question_length": len(question),
                        "llm_params": llm_params
                    }
                )
                
                result = loop.run_until_complete(
                    self.llm_service.generate_response(
                        question=question,
                        model_name=model_name,
                        model_path=model_path,
                        template=template,
                        **llm_params
                    )
                )
                
                if result.get('success'):
                    # Log successful generation
                    log_custom_event(
                        "text_generation_success",
                        f"Text generation completed successfully with model: {model_name}",
                        {
                            "model_name": model_name,
                            "response_length": len(result.get('response', '')),
                            "generation_time": result.get('generation_time'),
                        }
                    )
                    return jsonify(result), 200
                else:
                    # Log generation failure
                    log_custom_event(
                        "text_generation_failure",
                        f"Text generation failed with model: {model_name}",
                        {
                            "model_name": model_name,
                            "error": result.get('error')
                        }
                    )
                    return jsonify(result), 500
                    
            finally:
                loop.close()
                
        except Exception as e:
            log_custom_event(
                "text_generation_error",
                f"Text generation error with model {model_name}: {str(e)}",
                {
                    "model_name": model_name,
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            logger.error(f"Error generating response: {e}")
            return jsonify({
                "error": f"Failed to generate response: {str(e)}",
                "success": False
            }), 500
    
    def health_check(self):
        """Handle health check requests."""
        return jsonify({
            "status": "healthy",
            "service": "Local LLM API",
            "version": "1.0.0"
        }), 200
    
    @log_endpoint("clear_cache")
    def clear_cache(self):
        """Handle cache clear requests."""
        try:
            self.llm_service.clear_cache()
            
            # Log cache clear
            log_custom_event(
                "cache_cleared",
                "LLM cache cleared successfully",
                {}
            )
            
            return jsonify({"message": "Cache cleared successfully"}), 200
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return jsonify({"error": f"Failed to clear cache: {str(e)}"}), 500
    
    def cache_status(self):
        """Handle cache status requests."""
        try:
            cached_models = self.llm_service.get_cached_models()
            
            return jsonify({
                "cached_models": cached_models,
                "count": len(cached_models)
            }), 200
        except Exception as e:
            logger.error(f"Error getting cache status: {e}")
            return jsonify({"error": f"Failed to get cache status: {str(e)}"}), 500
    
    def get_logs(self):
        """Return all API and error logs (last 100 lines each)."""
        from utils.log_viewer import LogViewer
        log_viewer = LogViewer(self.app_config.LOG_DIR if hasattr(self.app_config, 'LOG_DIR') else './logs')
        api_logs = log_viewer.get_api_logs()
        error_logs = log_viewer.get_error_logs()
        return jsonify({
            "api_logs": api_logs,
            "error_logs": error_logs
        }), 200
    
    @log_endpoint("clear_logs")
    def clear_logs(self):
        """Handle clear logs requests."""
        try:
            from utils.log_viewer import LogViewer
            log_viewer = LogViewer(self.app_config.LOG_DIR if hasattr(self.app_config, 'LOG_DIR') else './logs')
            log_viewer.clear_logs()
            return jsonify({"message": "Logs cleared successfully"}), 200
        except Exception as e:
            logger.error(f"Error clearing logs: {e}")
            return jsonify({"error": f"Failed to clear logs: {str(e)}"}), 500


