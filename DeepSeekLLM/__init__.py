"""
DeepSeek LLM module for Local LLM API.
"""
from .deepseek_service import DeepSeekService
from .deepseek_routes import create_deepseek_routes
from .services import generate_text

__all__ = ['DeepSeekService', 'create_deepseek_routes', 'generate_text']
