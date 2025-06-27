"""
Configuration settings for the Local LLM application.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Flask settings
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', './models')
    MODELS_JSON_FILE = os.environ.get('MODELS_JSON_FILE', './models/models_list.json')
    ALLOWED_EXTENSIONS = {'gguf'}
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 10 * 1024 * 1024 * 1024))
    
   
    # LLM settings
    DEFAULT_N_GPU_LAYERS = int(os.environ.get('DEFAULT_N_GPU_LAYERS', 40))
    DEFAULT_N_BATCH = int(os.environ.get('DEFAULT_N_BATCH', 512))
    DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', 0.6))
    DEFAULT_VERBOSE = os.environ.get('DEFAULT_VERBOSE', 'True').lower() == 'true'
    
    # DeepSeek specific settings for large context handling
    DEEPSEEK_MAX_CONTEXT = int(os.environ.get('DEEPSEEK_MAX_CONTEXT', 32768))  # Maximum context size
    DEEPSEEK_DEFAULT_CONTEXT = int(os.environ.get('DEEPSEEK_DEFAULT_CONTEXT', 8192))  # Default context size
    DEEPSEEK_CHUNK_SIZE = int(os.environ.get('DEEPSEEK_CHUNK_SIZE', 4096))  # Chunk size for large inputs
    DEEPSEEK_OVERLAP_SIZE = int(os.environ.get('DEEPSEEK_OVERLAP_SIZE', 256))  # Overlap between chunks
    DEEPSEEK_MAX_THREADS = int(os.environ.get('DEEPSEEK_MAX_THREADS', 16))  # Maximum CPU threads
    DEEPSEEK_USE_MMAP = os.environ.get('DEEPSEEK_USE_MMAP', 'True').lower() == 'true'  # Use memory mapping
    DEEPSEEK_STREAMING_ENABLED = os.environ.get('DEEPSEEK_STREAMING_ENABLED', 'True').lower() == 'true'  # Enable streaming
    DEEPSEEK_CACHE_ENABLED = os.environ.get('DEEPSEEK_CACHE_ENABLED', 'True').lower() == 'true'  # Enable model caching
    
    # API settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
    API_PREFIX = os.environ.get('API_PREFIX', '/api/v1')
    
    # Logging settings
    LOG_DIR = os.environ.get('LOG_DIR', './logs')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', 10 * 1024 * 1024))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', 5))
    LOG_API_CALLS = os.environ.get('LOG_API_CALLS', 'True').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
