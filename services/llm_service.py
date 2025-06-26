"""
LLM service for handling language model operations.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

logger = logging.getLogger(__name__)

class LLMService:
    """Service for handling LLM operations."""
    
    def __init__(self, config):
        self.config = config
        self._llm_cache = {}  # Cache for loaded models
    
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
    
    async def generate_response(
        self, 
        question: str, 
        model_name: str, 
        model_path: str,
        template: Optional[str] = None,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """Generate a response using the specified model."""
        try:
            # Use default template if none provided
            if template is None:
                template = "Question: {question}\n\nAnswer: "
            
            # Create prompt template
            prompt = PromptTemplate(
                template=template, 
                input_variables=["question"]
            )
            
            # Create or get cached LLM
            cache_key = f"{model_name}_{hash(str(llm_kwargs))}"
            if cache_key not in self._llm_cache:
                self._llm_cache[cache_key] = self._create_llm(model_path, **llm_kwargs)
            
            llm = self._llm_cache[cache_key]
            
            # Create LLM chain
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            
            # Generate response
            logger.info(f"Generating response for question: {question[:50]}...")
            result = await llm_chain.ainvoke({"question": question})
            
            logger.info("Response generated successfully")
            return {
                "success": True,
                "response": result,
                "model_used": model_name,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": model_name,
                "question": question
            }
    
    def clear_cache(self) -> None:
        """Clear the LLM cache."""
        self._llm_cache.clear()
        logger.info("LLM cache cleared")
    
    def get_cached_models(self) -> list:
        """Get list of currently cached models."""
        return list(self._llm_cache.keys())
