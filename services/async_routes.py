"""
Async route handler wrapper for Flask with concurrency support.
"""
import asyncio
import logging
from functools import wraps
from typing import Callable, Any
from flask import current_app, request, jsonify
import time

logger = logging.getLogger(__name__)

def async_route(f: Callable) -> Callable:
    """Decorator to handle async routes in Flask."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Get the event loop for this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async function
        return loop.run_until_complete(f(*args, **kwargs))
    
    return wrapper

def concurrent_route(priority: int = 1, timeout: int = None):
    """Decorator to handle routes with concurrency management."""
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        @async_route
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Get concurrency manager from app
            concurrency_manager = getattr(current_app, 'concurrency_manager', None)
            
            if concurrency_manager is None:
                # Fallback to direct execution if no concurrency manager
                logger.warning("No concurrency manager found, executing directly")
                return await f(*args, **kwargs)
            
            try:
                # Submit request to concurrency manager
                request_id = concurrency_manager.submit_request(
                    func=lambda: asyncio.run(f(*args, **kwargs)),
                    priority=priority,
                    timeout=timeout or current_app.config.get('REQUEST_TIMEOUT', 300)
                )
                
                # Wait for result
                result = await concurrency_manager.wait_for_result(
                    request_id,
                    timeout=timeout or current_app.config.get('REQUEST_TIMEOUT', 300)
                )
                
                # Add timing information
                if isinstance(result, dict):
                    result['processing_time'] = time.time() - start_time
                    result['request_id'] = request_id
                
                return result
                
            except TimeoutError as e:
                logger.error(f"Request timeout: {e}")
                return jsonify({
                    'success': False,
                    'error': 'Request timeout',
                    'processing_time': time.time() - start_time
                }), 408
                
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'processing_time': time.time() - start_time
                }), 500
        
        return wrapper
    return decorator

def rate_limited_route(max_requests_per_minute: int = 60):
    """Decorator to add basic rate limiting."""
    request_times = {}
    
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old entries
            if client_ip in request_times:
                request_times[client_ip] = [
                    req_time for req_time in request_times[client_ip]
                    if current_time - req_time < 60
                ]
            else:
                request_times[client_ip] = []
            
            # Check rate limit
            if len(request_times[client_ip]) >= max_requests_per_minute:
                return jsonify({
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'retry_after': 60
                }), 429
            
            # Add current request
            request_times[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        
        return wrapper
    return decorator
