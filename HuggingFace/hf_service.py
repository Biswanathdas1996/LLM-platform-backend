"""
Hugging Face service for handling HF model operations and text generation.
"""
import asyncio
import logging
import contextlib
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        AutoModelForSeq2SeqLM,
        pipeline,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class HuggingFaceService:
    """Service for handling Hugging Face model operations."""
    
    def __init__(self, config, hf_model_manager):
        self.config = config
        self.hf_model_manager = hf_model_manager
        self._model_cache = {}  # Cache for loaded models and tokenizers
        self._pipeline_cache = {}  # Cache for pipelines
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. HuggingFace functionality will be limited.")
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. GPU acceleration will not be available.")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        return {
            'transformers': TRANSFORMERS_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'cuda_available': TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False
        }
    
    def _get_device(self) -> str:
        """Get the appropriate device for model loading."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    async def load_model(self, model_id: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a Hugging Face model and tokenizer.
        
        Args:
            model_id: The Hugging Face model identifier
            force_reload: Whether to force reload even if cached
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                "success": False,
                "error": "Transformers library not available. Please install: pip install transformers torch"
            }
        
        try:
            # Check if model exists in our registry
            model_info = self.hf_model_manager.get_hf_model(model_id)
            if not model_info:
                return {
                    "success": False,
                    "error": f"Model '{model_id}' not found in registry. Please add it first."
                }
            
            # Use cache if available and not forcing reload
            if model_id in self._model_cache and not force_reload:
                logger.info(f"Using cached model: {model_id}")
                return {
                    "success": True,
                    "message": f"Model '{model_id}' loaded from cache",
                    "model_info": model_info
                }
            
            device = self._get_device()
            logger.info(f"Loading model '{model_id}' on device: {device}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Determine model type and load appropriate model class
            model_type = model_info.get('model_type', 'text-generation')
            
            if model_type in ['text-generation', 'conversational']:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" and TORCH_AVAILABLE else None
                )
            elif model_type in ['text2text-generation', 'summarization', 'translation']:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    device_map="auto" if device == "cuda" else None,
                    torch_dtype=torch.float16 if device == "cuda" and TORCH_AVAILABLE else None
                )
            else:
                # For other types, try causal LM first
                try:
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                except:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            # Cache the model and tokenizer
            self._model_cache[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'model_info': model_info,
                'loaded_at': datetime.now().isoformat(),
                'device': device
            }
            
            logger.info(f"Successfully loaded model: {model_id}")
            return {
                "success": True,
                "message": f"Model '{model_id}' loaded successfully",
                "model_info": model_info,
                "device": device
            }
            
        except Exception as e:
            logger.error(f"Error loading model '{model_id}': {e}")
            return {
                "success": False,
                "error": f"Failed to load model: {str(e)}"
            }
    
    async def generate_response(
        self,
        model_id: str,
        prompt: str,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text response using a Hugging Face model.
        
        Args:
            model_id: The Hugging Face model identifier
            prompt: The input prompt
            **generation_kwargs: Additional generation parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                "success": False,
                "error": "Transformers library not available"
            }
        
        try:
            # Load model if not cached
            if model_id not in self._model_cache:
                load_result = await self.load_model(model_id)
                if not load_result["success"]:
                    return load_result
            
            cached_model = self._model_cache[model_id]
            model = cached_model['model']
            tokenizer = cached_model['tokenizer']
            model_info = cached_model['model_info']
            
            # Get default parameters from model config
            default_params = model_info.get('parameters', {})
            
            # Merge with provided parameters
            params = {
                'max_new_tokens': 100,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                **default_params,
                **generation_kwargs
            }
            
            # Tokenize input
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if cached_model['device'] == "cuda" and TORCH_AVAILABLE:
                inputs = inputs.to("cuda")
            
            # Generate response
            logger.info(f"Generating response with model: {model_id}")
            with torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext():
                outputs = model.generate(
                    inputs,
                    **params,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Update usage statistics
            self.hf_model_manager.update_model_usage(model_id)
            
            logger.info(f"Response generated successfully with model: {model_id}")
            return {
                "success": True,
                "response": response,
                "model_used": model_id,
                "prompt": prompt,
                "parameters": params,
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"Error generating response with model '{model_id}': {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": model_id,
                "prompt": prompt
            }
    
    async def generate_with_pipeline(
        self,
        model_id: str,
        task: str,
        inputs: Any,
        **pipeline_kwargs
    ) -> Dict[str, Any]:
        """
        Generate response using Hugging Face pipeline.
        
        Args:
            model_id: The Hugging Face model identifier
            task: The task type (e.g., 'text-generation', 'summarization')
            inputs: The input data
            **pipeline_kwargs: Additional pipeline parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                "success": False,
                "error": "Transformers library not available"
            }
        
        try:
            # Check if model exists in our registry
            model_info = self.hf_model_manager.get_hf_model(model_id)
            if not model_info:
                return {
                    "success": False,
                    "error": f"Model '{model_id}' not found in registry"
                }
            
            pipeline_key = f"{model_id}_{task}"
            
            # Create pipeline if not cached
            if pipeline_key not in self._pipeline_cache:
                device = 0 if self._get_device() == "cuda" else -1
                pipe = pipeline(
                    task,
                    model=model_id,
                    device=device,
                    **pipeline_kwargs
                )
                self._pipeline_cache[pipeline_key] = pipe
            
            pipe = self._pipeline_cache[pipeline_key]
            
            # Generate response
            logger.info(f"Generating with pipeline: {model_id} - {task}")
            results = pipe(inputs)
            
            # Update usage statistics
            self.hf_model_manager.update_model_usage(model_id)
            
            return {
                "success": True,
                "results": results,
                "model_used": model_id,
                "task": task,
                "inputs": inputs,
                "model_info": model_info
            }
            
        except Exception as e:
            logger.error(f"Error with pipeline '{model_id}' - '{task}': {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": model_id,
                "task": task
            }
    
    async def download_gguf_model(
        self, 
        model_id: str, 
        filename: Optional[str] = None,
        download_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Download GGUF format model from HuggingFace Hub.
        
        Args:
            model_id: The HuggingFace model identifier
            filename: Specific GGUF file to download (optional)
            download_path: Path to save the model (optional, uses config default)
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            HF_HUB_AVAILABLE = True
        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub library not available. Please install: pip install huggingface_hub"
            }
        
        try:
            # Check if model exists in our registry
            model_info = self.hf_model_manager.get_hf_model(model_id)
            if not model_info:
                return {
                    "success": False,
                    "error": f"Model '{model_id}' not found in registry. Please add it first."
                }
            
            # Set download path
            if not download_path:
                download_path = self.config.UPLOAD_FOLDER
            
            # List available files in the repository
            try:
                repo_files = list_repo_files(model_id)
                gguf_files = [f for f in repo_files if f.endswith('.gguf')]
                
                if not gguf_files:
                    return {
                        "success": False,
                        "error": f"No GGUF files found in model '{model_id}'"
                    }
                
                # If no specific filename provided, use the first GGUF file
                if not filename:
                    filename = gguf_files[0]
                elif filename not in gguf_files:
                    return {
                        "success": False,
                        "error": f"GGUF file '{filename}' not found. Available: {gguf_files}"
                    }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to list repository files: {str(e)}"
                }
            
            # Download the GGUF file
            logger.info(f"Downloading GGUF model: {model_id}/{filename}")
            
            try:
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=download_path,
                    local_dir_use_symlinks=False
                )
                
                # Update model info with download details
                download_info = {
                    "downloaded_at": datetime.now().isoformat(),
                    "gguf_file": filename,
                    "local_path": downloaded_path,
                    "available_gguf_files": gguf_files
                }
                
                logger.info(f"Successfully downloaded GGUF model: {downloaded_path}")
                return {
                    "success": True,
                    "message": f"GGUF model '{filename}' downloaded successfully",
                    "model_info": model_info,
                    "download_info": download_info
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to download GGUF file: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error downloading GGUF model '{model_id}': {e}")
            return {
                "success": False,
                "error": f"Failed to download GGUF model: {str(e)}"
            }
    
    async def list_gguf_files(self, model_id: str) -> Dict[str, Any]:
        """
        List available GGUF files for a model on HuggingFace Hub.
        
        Args:
            model_id: The HuggingFace model identifier
        """
        try:
            from huggingface_hub import list_repo_files
        except ImportError:
            return {
                "success": False,
                "error": "huggingface_hub library not available. Please install: pip install huggingface_hub"
            }
        
        try:
            repo_files = list_repo_files(model_id)
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            
            return {
                "success": True,
                "model_id": model_id,
                "gguf_files": gguf_files,
                "total_gguf_files": len(gguf_files)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list GGUF files: {str(e)}"
            }
    
    def clear_cache(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear model cache."""
        if model_id:
            if model_id in self._model_cache:
                del self._model_cache[model_id]
                # Clear related pipelines
                keys_to_remove = [k for k in self._pipeline_cache.keys() if k.startswith(model_id)]
                for key in keys_to_remove:
                    del self._pipeline_cache[key]
                logger.info(f"Cleared cache for model: {model_id}")
                return {"success": True, "message": f"Cache cleared for {model_id}"}
            else:
                return {"success": False, "message": f"Model {model_id} not in cache"}
        else:
            self._model_cache.clear()
            self._pipeline_cache.clear()
            logger.info("Cleared all model caches")
            return {"success": True, "message": "All caches cleared"}
    
    def get_cached_models(self) -> List[str]:
        """Get list of currently cached models."""
        return list(self._model_cache.keys())
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        cache_info = {}
        for model_id, cached_data in self._model_cache.items():
            cache_info[model_id] = {
                'loaded_at': cached_data.get('loaded_at'),
                'device': cached_data.get('device'),
                'model_type': cached_data.get('model_info', {}).get('model_type')
            }
        
        # Get HuggingFace cache directory
        hf_cache_dir = os.environ.get('HF_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'))
        transformers_cache = os.path.join(hf_cache_dir, 'transformers')
        
        return {
            'cached_models': cache_info,
            'cached_pipelines': list(self._pipeline_cache.keys()),
            'total_cached': len(self._model_cache),
            'cache_directories': {
                'hf_cache_dir': hf_cache_dir,
                'transformers_cache': transformers_cache,
                'cache_exists': os.path.exists(transformers_cache)
            }
        }
