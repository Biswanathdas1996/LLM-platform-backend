"""
Utility functions for the Local LLM application.
"""
import os
import hashlib
from typing import Optional

def get_file_hash(file_path: str, algorithm: str = 'md5') -> Optional[str]:
    """Calculate hash of a file."""
    if not os.path.exists(file_path):
        return None
    
    hash_func = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except (OSError, IOError):
        return None

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing potentially dangerous characters."""
    # Remove or replace dangerous characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing whitespace and dots
    filename = filename.strip(' .')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed_file'
    
    return filename

def validate_model_name(model_name: str) -> bool:
    """Validate model name format."""
    if not model_name:
        return False
    
    # Check for valid characters (alphanumeric, hyphen, underscore, dot)
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.')
    return all(c in allowed_chars for c in model_name)

def get_available_memory() -> Optional[int]:
    """Get available system memory in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None
