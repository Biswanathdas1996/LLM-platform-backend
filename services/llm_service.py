"""
LLM service for handling language model operations with concurrency support.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from .model_pool import ModelPoolManager, ModelInstance

logger = logging.getLogger(__name__)

class LLMService:
    """Service for handling LLM operations with connection pooling and concurrency."""
    
    def __init__(self, config):
        self.config = config
        self.pool_manager = ModelPoolManager(config)
        self._generation_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'concurrent_requests': 0
        }
    
    def _create_llm(self, model_path: str, **kwargs) -> LlamaCpp:
        """Create a LlamaCpp instance with given parameters."""
        default_params = {
            'n_gpu_layers': self.config.DEFAULT_N_GPU_LAYERS,
            'n_batch': self.config.DEFAULT_N_BATCH,
            'verbose': self.config.DEFAULT_VERBOSE,
            'callback_manager': CallbackManager([StreamingStdOutCallbackHandler()])
        }
        
        # Override defaults with provided kwargs
        params = {**default_params, **kwargs}
        params['model_path'] = model_path
        
        logger.info(f"Creating LLM with model: {model_path}")
        return LlamaCpp(**params)
    
    def _execute_generation(self, instance: ModelInstance, prompt_template: str, 
                          question: str) -> Dict[str, Any]:
        """Execute the actual text generation using a model instance."""
        try:
            # Create prompt template
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["question"]
            )
            
            # Create LLM chain
            llm_chain = LLMChain(prompt=prompt, llm=instance.model)
            
            # Generate response synchronously
            result = llm_chain.invoke({"question": question})
            
            return {
                "success": True,
                "response": result,
                "model_path": instance.model_path,
                "question": question,
                "usage_count": instance.usage_count
            }
            
        except Exception as e:
            logger.error(f"Error in generation execution: {e}")
            raise
    
    async def generate_response(
        self, 
        question: str, 
        model_name: str, 
        model_path: str,
        template: Optional[str] = None,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """Generate a response using the specified model with connection pooling."""
        
        self._generation_stats['total_requests'] += 1
        self._generation_stats['concurrent_requests'] += 1
        
        try:
            # Use default template if none provided
            if template is None:
                template = "Question: {question}\n\nAnswer: "
            
            logger.info(f"Generating response for question: {question[:50]}...")
            
            # Get model instance from pool
            instance = self.pool_manager.get_model_instance(
                model_path=model_path,
                create_func=self._create_llm,
                timeout=self.config.QUEUE_TIMEOUT,
                **llm_kwargs
            )
            
            if instance is None:
                raise Exception("Failed to acquire model instance from pool")
            
            try:
                # Execute generation in a separate thread to avoid blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self._execute_generation,
                    instance, 
                    template, 
                    question
                )
                
                result.update({
                    "model_used": model_name,
                    "pool_stats": self.pool_manager.get_all_stats()
                })
                
                self._generation_stats['successful_requests'] += 1
                logger.info("Response generated successfully")
                return result
                
            finally:
                # Always return instance to pool
                self.pool_manager.return_model_instance(instance)
                
        except Exception as e:
            self._generation_stats['failed_requests'] += 1
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": model_name,
                "question": question
            }
        finally:
            self._generation_stats['concurrent_requests'] -= 1
    
    def clear_cache(self) -> None:
        """Clear all model pools."""
        self.pool_manager.clear_all_pools()
        logger.info("All model pools cleared")
    
    def get_cached_models(self) -> list:
        """Get list of currently pooled models."""
        stats = self.pool_manager.get_all_stats()
        return list(stats['pools'].keys())
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        pool_stats = self.pool_manager.get_all_stats()
        
        return {
            'generation_stats': self._generation_stats.copy(),
            'pool_stats': pool_stats,
            'service_status': {
                'active': True,
                'total_pools': pool_stats['summary']['total_pools'],
                'total_instances': pool_stats['summary']['total_instances']
            }
        }
