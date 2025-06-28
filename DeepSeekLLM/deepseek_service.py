"""
DeepSeek LLM Service for API integration with concurrency support.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional, List
from threading import Lock
from .services import generate_text

logger = logging.getLogger(__name__)

class DeepSeekService:
    """Service class for DeepSeek LLM operations."""
    
    def __init__(self, config):
        """Initialize the DeepSeek service."""
        self.config = config
        self.models_cache = {}
        self.cache_lock = Lock()
        self.models_dir = os.path.join(os.path.dirname(__file__), "model")
        
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available DeepSeek models."""
        models = []
        
        if not os.path.exists(self.models_dir):
            logger.warning(f"DeepSeek models directory not found: {self.models_dir}")
            return models
            
        try:
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.gguf'):
                    model_path = os.path.join(self.models_dir, filename)
                    file_size = os.path.getsize(model_path)
                    
                    models.append({
                        "name": filename,
                        "path": model_path,
                        "size": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "type": "deepseek"
                    })
                    
        except Exception as e:
            logger.error(f"Error listing DeepSeek models: {e}")
            
        return models
    
    def generate_response(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        n_ctx: int = 2048,
        n_threads: int = 8,
        use_mlock: bool = True,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate text using a DeepSeek model (thread-safe)."""
        
        try:
            # Find the model file
            model_path = self._get_model_path(model_name)
            if not model_path:
                raise ValueError(f"Model '{model_name}' not found")
            
            # Set default stop sequences if none provided
            if stop is None:
                stop = ["</s>", "<|end|>", "<|endoftext|>"]
            
            logger.info(f"Generating text with DeepSeek model: {model_name}")
            logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            
            # Use cache lock to ensure thread safety
            with self.cache_lock:
                # Generate text using the services module
                generated_text = generate_text(
                    model_path=model_path,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    use_mlock=use_mlock,
                    stop=stop
                )
            
            return {
                "success": True,
                "model": model_name,
                "prompt": prompt,
                "generated_text": generated_text,
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "n_ctx": n_ctx,
                    "n_threads": n_threads,
                    "stop": stop
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating DeepSeek response: {e}")
            return {
                "success": False,
                "error": str(e),
                "model": model_name,
                "prompt": prompt
            }
    
    async def generate_response_async(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        n_ctx: int = 2048,
        n_threads: int = 8,
        use_mlock: bool = True,
        stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Async wrapper for generate_response to avoid blocking."""
        
        # Run the synchronous generation in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_response,
            model_name,
            prompt,
            max_tokens,
            temperature,
            n_ctx,
            n_threads,
            use_mlock,
            stop
        )
    
    def _get_model_path(self, model_name: str) -> Optional[str]:
        """Get the full path to a model file."""
        # Check if it's already a full path
        if os.path.isfile(model_name):
            return model_name
            
        # Check in the models directory
        model_path = os.path.join(self.models_dir, model_name)
        if os.path.isfile(model_path):
            return model_path
            
        # Check without .gguf extension
        if not model_name.endswith('.gguf'):
            model_path_with_ext = os.path.join(self.models_dir, f"{model_name}.gguf")
            if os.path.isfile(model_path_with_ext):
                return model_path_with_ext
                
        return None
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        model_path = self._get_model_path(model_name)
        
        if not model_path:
            return {
                "success": False,
                "error": f"Model '{model_name}' not found"
            }
            
        try:
            file_size = os.path.getsize(model_path)
            return {
                "success": True,
                "name": model_name,
                "path": model_path,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2),
                "type": "deepseek",
                "exists": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting model info: {e}"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the DeepSeek service."""
        try:
            models = self.get_available_models()
            models_dir_exists = os.path.exists(self.models_dir)
            
            return {
                "success": True,
                "service": "DeepSeek LLM Service",
                "status": "healthy",
                "models_available": len(models),
                "models_directory": self.models_dir,
                "models_directory_exists": models_dir_exists,
                "models": [model["name"] for model in models]
            }
        except Exception as e:
            return {
                "success": False,
                "service": "DeepSeek LLM Service", 
                "status": "unhealthy",
                "error": str(e)
            }
