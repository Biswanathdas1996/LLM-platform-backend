"""
Logging utilities for the Local LLM application.
Provides structured logging for API calls, responses, and errors.
"""
import os
import json
import logging
import logging.handlers
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import request, g
import time

class APILogFormatter(logging.Formatter):
    """Custom formatter for API logs with structured output."""
    
    def format(self, record):
        """Format log record with additional API context."""
        # Get the original message
        msg = super().format(record)
        
        # Create IST timezone (UTC+5:30)
        ist_timezone = timezone(timedelta(hours=5, minutes=30))
        
        # Add API-specific fields if available
        if hasattr(record, 'api_data'):
            api_data = record.api_data
            formatted_data = {
                'timestamp': datetime.now(ist_timezone).isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                **api_data
            }
            return json.dumps(formatted_data, indent=2)
        
        return msg

class APILogger:
    """Handles API-specific logging operations."""
    
    def __init__(self, app=None):
        self.app = app
        self.logger = None
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize logging for the Flask app."""
        self.app = app
        self.setup_loggers(app.config)
        
        # Register before and after request handlers
        app.before_request(self._before_request)
        app.after_request(self._after_request)
        app.teardown_appcontext(self._teardown)
    
    def setup_loggers(self, config):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = config.get('LOG_DIR', './logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure main API logger
        self.logger = logging.getLogger('api_logger')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler for API logs
        api_log_file = os.path.join(log_dir, 'api.log')
        file_handler = logging.handlers.RotatingFileHandler(
            api_log_file,
            maxBytes=config.get('LOG_MAX_BYTES', 10 * 1024 * 1024),  # 10MB
            backupCount=config.get('LOG_BACKUP_COUNT', 5)
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(APILogFormatter())
        
        # Console handler for development
        if config.get('DEBUG', False):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(APILogFormatter())
            self.logger.addHandler(console_handler)
        
        self.logger.addHandler(file_handler)
        
        # Error logger
        self.error_logger = logging.getLogger('error_logger')
        self.error_logger.setLevel(logging.ERROR)
        
        # Remove existing handlers
        self.error_logger.handlers.clear()
        
        # Error file handler
        error_log_file = os.path.join(log_dir, 'errors.log')
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=config.get('LOG_MAX_BYTES', 10 * 1024 * 1024),
            backupCount=config.get('LOG_BACKUP_COUNT', 5)
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(APILogFormatter())
        self.error_logger.addHandler(error_file_handler)
    
    def _before_request(self):
        """Log incoming requests."""
        g.start_time = time.time()
        g.request_id = self._generate_request_id()
        
        # Skip logging for health checks and static files
        if self._should_skip_logging():
            return
        
        api_data = {
            'request_id': g.request_id,
            'method': request.method,
            'url': request.url,
            'endpoint': request.endpoint,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'content_type': request.content_type,
            'content_length': request.content_length,
            'args': dict(request.args),
            'type': 'request'
        }
        
        # Log request body for POST/PUT requests (but not file uploads)
        if request.method in ['POST', 'PUT'] and request.is_json:
            try:
                api_data['request_body'] = request.get_json()
            except Exception:
                api_data['request_body'] = 'Unable to parse JSON'
        
        self.log_api_call("Incoming request", api_data)
    
    def _after_request(self, response):
        """Log outgoing responses."""
        if self._should_skip_logging():
            return response
        
        duration = time.time() - getattr(g, 'start_time', 0)
        
        api_data = {
            'request_id': getattr(g, 'request_id', 'unknown'),
            'method': request.method,
            'url': request.url,
            'endpoint': request.endpoint,
            'status_code': response.status_code,
            'content_type': response.content_type,
            'content_length': response.content_length,
            'duration_ms': round(duration * 1000, 2),
            'type': 'response'
        }
        
        # Log response body for JSON responses (limit size)
        if response.is_json and response.content_length and response.content_length < 10000:
            try:
                api_data['response_body'] = response.get_json()
            except Exception:
                api_data['response_body'] = 'Unable to parse JSON'
        
        level = 'INFO'
        message = "Outgoing response"
        
        # Log as error for 4xx/5xx status codes
        if response.status_code >= 400:
            level = 'ERROR'
            message = f"Error response - {response.status_code}"
        
        if level == 'ERROR':
            self.log_error(message, api_data)
        else:
            self.log_api_call(message, api_data)
        
        return response
    
    def _teardown(self, exception):
        """Handle request teardown and log any exceptions."""
        if exception is not None:
            api_data = {
                'request_id': getattr(g, 'request_id', 'unknown'),
                'method': request.method,
                'url': request.url,
                'endpoint': request.endpoint,
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'type': 'exception'
            }
            self.log_error("Unhandled exception during request", api_data, exc_info=True)
    
    def _should_skip_logging(self):
        """Determine if logging should be skipped for this request."""
        # Skip health checks, logs endpoint, cache status, and static files
        skip_endpoints = ['api.health_check', 'api.get_logs', 'api.clear_logs', 'api.cache_status', 'health_check', 'static']
        skip_paths = ['/favicon.ico']
        
        return (
            request.endpoint in skip_endpoints or
            request.path in skip_paths or
            request.path.startswith('/static/')
        )
    
    def _generate_request_id(self):
        """Generate a unique request ID."""
        return f"req_{int(time.time() * 1000000)}"
    
    def log_api_call(self, message, api_data=None):
        """Log an API call with structured data."""
        if self.logger:
            record = self.logger.makeRecord(
                self.logger.name,
                logging.INFO,
                __file__,
                0,
                message,
                (),
                None
            )
            record.api_data = api_data or {}
            self.logger.handle(record)
    
    def log_error(self, message, api_data=None, exc_info=False):
        """Log an error with structured data."""
        if self.error_logger:
            record = self.error_logger.makeRecord(
                self.error_logger.name,
                logging.ERROR,
                __file__,
                0,
                message,
                (),
                None,
                exc_info=exc_info
            )
            record.api_data = api_data or {}
            self.error_logger.handle(record)

def log_endpoint(endpoint_name):
    """Decorator to add additional logging to specific endpoints."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = logging.getLogger('api_logger')
            
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"Endpoint {endpoint_name} completed successfully in {duration:.3f}s",
                    extra={'api_data': {
                        'endpoint': endpoint_name,
                        'duration_s': round(duration, 3),
                        'status': 'success',
                        'type': 'endpoint_execution'
                    }}
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_logger = logging.getLogger('error_logger')
                
                error_logger.error(
                    f"Endpoint {endpoint_name} failed after {duration:.3f}s: {str(e)}",
                    extra={'api_data': {
                        'endpoint': endpoint_name,
                        'duration_s': round(duration, 3),
                        'status': 'error',
                        'error_message': str(e),
                        'error_type': type(e).__name__,
                        'type': 'endpoint_execution'
                    }},
                    exc_info=True
                )
                
                raise
        
        return wrapper
    return decorator

# Convenience function for manual logging
def log_custom_event(event_type, message, data=None):
    """Log a custom event with structured data."""
    logger = logging.getLogger('api_logger')
    api_data = {
        'event_type': event_type,
        'type': 'custom_event',
        **(data or {})
    }
    
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        (),
        None
    )
    record.api_data = api_data
    logger.handle(record)
