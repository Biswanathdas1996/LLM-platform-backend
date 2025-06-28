"""
Model pool manager for efficient LLM instance management.
"""
import threading
import logging
from typing import Dict, Any, Optional, List
from queue import Queue, Empty
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ModelInstance:
    """Represents a model instance in the pool."""
    model: Any
    model_path: str
    config_hash: str
    created_at: datetime
    last_used: datetime
    usage_count: int = 0
    is_busy: bool = False
    
    def mark_used(self):
        """Mark the model as used."""
        self.last_used = datetime.now()
        self.usage_count += 1

class ModelPool:
    """Thread-safe pool of model instances."""
    
    def __init__(self, model_path: str, create_model_func: callable, 
                 max_instances: int = 3, max_idle_time: int = 300):
        self.model_path = model_path
        self.create_model_func = create_model_func
        self.max_instances = max_instances
        self.max_idle_time = max_idle_time
        
        self.instances: List[ModelInstance] = []
        self.available_queue = Queue(maxsize=max_instances)
        self.lock = threading.RLock()
        
        self.total_requests = 0
        self.cache_hits = 0
        
    def get_instance(self, config_hash: str, timeout: int = 30) -> Optional[ModelInstance]:
        """Get an available model instance or create a new one."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to get an available instance
                instance = self.available_queue.get_nowait()
                
                with self.lock:
                    if instance.config_hash == config_hash and not instance.is_busy:
                        instance.is_busy = True
                        instance.mark_used()
                        self.cache_hits += 1
                        self.total_requests += 1
                        logger.debug(f"Reusing model instance for {self.model_path}")
                        return instance
                    else:
                        # Config mismatch, put it back
                        self.available_queue.put(instance)
                        
            except Empty:
                pass
            
            # Try to create a new instance if under limit
            with self.lock:
                if len(self.instances) < self.max_instances:
                    try:
                        model = self.create_model_func()
                        instance = ModelInstance(
                            model=model,
                            model_path=self.model_path,
                            config_hash=config_hash,
                            created_at=datetime.now(),
                            last_used=datetime.now(),
                            is_busy=True
                        )
                        self.instances.append(instance)
                        self.total_requests += 1
                        logger.info(f"Created new model instance for {self.model_path}")
                        return instance
                        
                    except Exception as e:
                        logger.error(f"Failed to create model instance: {e}")
                        return None
            
            # Wait a bit and retry
            time.sleep(0.1)
        
        logger.warning(f"Timeout waiting for model instance: {self.model_path}")
        return None
    
    def return_instance(self, instance: ModelInstance):
        """Return an instance to the pool."""
        with self.lock:
            instance.is_busy = False
            self.available_queue.put(instance)
            logger.debug(f"Returned model instance to pool: {self.model_path}")
    
    def cleanup_idle_instances(self):
        """Remove instances that have been idle for too long."""
        current_time = datetime.now()
        
        with self.lock:
            idle_instances = [
                inst for inst in self.instances
                if not inst.is_busy and 
                (current_time - inst.last_used).seconds > self.max_idle_time
            ]
            
            for instance in idle_instances:
                try:
                    # Remove from available queue if present
                    temp_queue = Queue()
                    while not self.available_queue.empty():
                        item = self.available_queue.get_nowait()
                        if item != instance:
                            temp_queue.put(item)
                    
                    # Put remaining items back
                    while not temp_queue.empty():
                        self.available_queue.put(temp_queue.get_nowait())
                    
                    # Remove from instances list
                    self.instances.remove(instance)
                    
                    logger.info(f"Removed idle model instance: {self.model_path}")
                    
                except Exception as e:
                    logger.error(f"Error cleaning up idle instance: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            busy_count = sum(1 for inst in self.instances if inst.is_busy)
            
            return {
                'model_path': self.model_path,
                'total_instances': len(self.instances),
                'busy_instances': busy_count,
                'available_instances': len(self.instances) - busy_count,
                'max_instances': self.max_instances,
                'total_requests': self.total_requests,
                'cache_hits': self.cache_hits,
                'cache_hit_rate': self.cache_hits / max(self.total_requests, 1) * 100
            }

class ModelPoolManager:
    """Manages multiple model pools for different models."""
    
    def __init__(self, config):
        self.config = config
        self.pools: Dict[str, ModelPool] = {}
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("ModelPoolManager initialized")
    
    def _generate_config_hash(self, **kwargs) -> str:
        """Generate a hash for the model configuration."""
        config_str = str(sorted(kwargs.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_model_instance(self, model_path: str, create_func: callable, 
                          timeout: int = 30, **model_kwargs) -> Optional[ModelInstance]:
        """Get a model instance from the appropriate pool."""
        
        config_hash = self._generate_config_hash(**model_kwargs)
        
        with self.lock:
            # Get or create pool for this model
            if model_path not in self.pools:
                self.pools[model_path] = ModelPool(
                    model_path=model_path,
                    create_model_func=lambda: create_func(model_path, **model_kwargs),
                    max_instances=self.config.MODEL_POOL_SIZE,
                    max_idle_time=300
                )
                logger.info(f"Created new model pool for: {model_path}")
            
            pool = self.pools[model_path]
        
        return pool.get_instance(config_hash, timeout)
    
    def return_model_instance(self, instance: ModelInstance):
        """Return a model instance to its pool."""
        with self.lock:
            if instance.model_path in self.pools:
                self.pools[instance.model_path].return_instance(instance)
    
    def _cleanup_worker(self):
        """Background worker to clean up idle instances."""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                
                with self.lock:
                    for pool in self.pools.values():
                        pool.cleanup_idle_instances()
                        
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all pools."""
        with self.lock:
            pool_stats = {}
            total_instances = 0
            total_requests = 0
            total_cache_hits = 0
            
            for model_path, pool in self.pools.items():
                stats = pool.get_stats()
                pool_stats[model_path] = stats
                total_instances += stats['total_instances']
                total_requests += stats['total_requests']
                total_cache_hits += stats['cache_hits']
            
            return {
                'pools': pool_stats,
                'summary': {
                    'total_pools': len(self.pools),
                    'total_instances': total_instances,
                    'total_requests': total_requests,
                    'total_cache_hits': total_cache_hits,
                    'overall_cache_hit_rate': total_cache_hits / max(total_requests, 1) * 100
                }
            }
    
    def clear_all_pools(self):
        """Clear all model pools."""
        with self.lock:
            self.pools.clear()
            logger.info("All model pools cleared")
