"""
Concurrency manager for handling multiple simultaneous requests.
"""
import asyncio
import threading
import logging
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import uuid

logger = logging.getLogger(__name__)

@dataclass
class RequestTask:
    """Represents a request task in the queue."""
    id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 1
    created_at: datetime = None
    timeout: int = 300
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class ConcurrencyManager:
    """Manages concurrent request processing with queuing and limits."""
    
    def __init__(self, config):
        self.config = config
        self.max_concurrent = config.MAX_CONCURRENT_REQUESTS
        self.max_workers = config.MAX_WORKERS
        self.request_timeout = config.REQUEST_TIMEOUT
        self.queue_timeout = config.QUEUE_TIMEOUT
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="LLM-Worker"
        )
        
        # Request queue and tracking
        self.request_queue = Queue(maxsize=self.max_concurrent * 2)
        self.active_requests = {}
        self.request_stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'queued_requests': 0,
            'active_requests': 0
        }
        
        # Thread-safe locks
        self.stats_lock = threading.Lock()
        self.active_lock = threading.Lock()
        
        # Start background worker
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"ConcurrencyManager initialized with {self.max_workers} workers, "
                   f"{self.max_concurrent} max concurrent requests")
    
    def submit_request(self, func: Callable, *args, priority: int = 1, 
                      timeout: int = None, **kwargs) -> str:
        """Submit a request for processing."""
        
        if timeout is None:
            timeout = self.request_timeout
            
        # Create unique request ID
        request_id = str(uuid.uuid4())
        
        # Create task
        task = RequestTask(
            id=request_id,
            function=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        try:
            # Add to queue with timeout
            self.request_queue.put(task, timeout=self.queue_timeout)
            
            with self.stats_lock:
                self.request_stats['total_requests'] += 1
                self.request_stats['queued_requests'] += 1
                
            logger.info(f"Request {request_id} queued for processing")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to queue request {request_id}: {e}")
            with self.stats_lock:
                self.request_stats['failed_requests'] += 1
            raise
    
    def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a specific request."""
        with self.active_lock:
            if request_id in self.active_requests:
                task_info = self.active_requests[request_id]
                return {
                    'status': task_info['status'],
                    'started_at': task_info['started_at'].isoformat(),
                    'progress': task_info.get('progress', 'unknown')
                }
        
        return {'status': 'not_found'}
    
    async def wait_for_result(self, request_id: str, timeout: int = None) -> Dict[str, Any]:
        """Wait for a request to complete and return the result."""
        if timeout is None:
            timeout = self.request_timeout
            
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.active_lock:
                if request_id in self.active_requests:
                    task_info = self.active_requests[request_id]
                    
                    if task_info['status'] == 'completed':
                        result = task_info['result']
                        # Clean up completed request
                        del self.active_requests[request_id]
                        return result
                    
                    elif task_info['status'] == 'failed':
                        error = task_info['error']
                        # Clean up failed request
                        del self.active_requests[request_id]
                        raise Exception(f"Request failed: {error}")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        # Timeout reached
        with self.active_lock:
            if request_id in self.active_requests:
                del self.active_requests[request_id]
        
        raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
    
    def _process_queue(self):
        """Background worker that processes the request queue."""
        logger.info("Request queue processor started")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = self.request_queue.get(timeout=1.0)
                
                # Update stats
                with self.stats_lock:
                    self.request_stats['queued_requests'] -= 1
                    self.request_stats['active_requests'] += 1
                
                # Process task in thread pool
                self._process_task(task)
                
            except Empty:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
    
    def _process_task(self, task: RequestTask):
        """Process a single task."""
        # Add to active requests
        with self.active_lock:
            self.active_requests[task.id] = {
                'status': 'running',
                'started_at': datetime.now(),
                'task': task
            }
        
        def execute_task():
            try:
                logger.info(f"Processing request {task.id}")
                
                # Execute the actual function
                result = task.function(*task.args, **task.kwargs)
                
                # Mark as completed
                with self.active_lock:
                    if task.id in self.active_requests:
                        self.active_requests[task.id].update({
                            'status': 'completed',
                            'result': result,
                            'completed_at': datetime.now()
                        })
                
                with self.stats_lock:
                    self.request_stats['completed_requests'] += 1
                    self.request_stats['active_requests'] -= 1
                
                logger.info(f"Request {task.id} completed successfully")
                
            except Exception as e:
                logger.error(f"Request {task.id} failed: {e}")
                
                # Mark as failed
                with self.active_lock:
                    if task.id in self.active_requests:
                        self.active_requests[task.id].update({
                            'status': 'failed',
                            'error': str(e),
                            'completed_at': datetime.now()
                        })
                
                with self.stats_lock:
                    self.request_stats['failed_requests'] += 1
                    self.request_stats['active_requests'] -= 1
        
        # Submit to thread pool
        self.executor.submit(execute_task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current concurrency statistics."""
        with self.stats_lock:
            stats = self.request_stats.copy()
        
        stats.update({
            'queue_size': self.request_queue.qsize(),
            'max_concurrent': self.max_concurrent,
            'max_workers': self.max_workers,
            'executor_running': not self.executor._shutdown
        })
        
        return stats
    
    def shutdown(self, wait: bool = True):
        """Shutdown the concurrency manager."""
        logger.info("Shutting down ConcurrencyManager")
        
        self.running = False
        
        if wait:
            # Wait for worker thread to finish
            self.worker_thread.join(timeout=5.0)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
        
        logger.info("ConcurrencyManager shutdown complete")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'running') and self.running:
            self.shutdown(wait=False)
