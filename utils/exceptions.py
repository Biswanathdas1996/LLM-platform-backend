"""
Custom exceptions for the Local LLM application.
"""

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class ModelNotFoundError(LLMError):
    """Raised when a requested model is not found."""
    pass

class ModelLoadError(LLMError):
    """Raised when a model fails to load."""
    pass

class InvalidModelError(LLMError):
    """Raised when a model file is invalid or corrupted."""
    pass

class GenerationError(LLMError):
    """Raised when text generation fails."""
    pass

class ConfigurationError(Exception):
    """Raised when there's a configuration error."""
    pass
