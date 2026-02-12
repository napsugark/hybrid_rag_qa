"""
Resilience utilities for production RAG system
Includes retry logic, circuit breaker, and error handling
"""

import time
import logging
from typing import Callable, Any, Optional, TypeVar, Dict
from functools import wraps
from collections import deque
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior"""
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def retry_with_backoff(
    config: RetryConfig,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator that retries a function with exponential backoff
    
    Args:
        config: RetryConfig instance
        exceptions: Tuple of exception types to catch
        on_retry: Optional callback function called on each retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import random
            
            delay = config.initial_delay
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {config.max_attempts} attempts: {e}"
                        )
                        raise
                    
                    # Calculate delay with jitter
                    current_delay = min(
                        delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )
                    if config.jitter:
                        current_delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{config.max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(current_delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "CircuitBreaker",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._state = self.CLOSED
        self._lock = threading.Lock()
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )
    
    @property
    def state(self) -> str:
        return self._state
    
    def _transition_to_open(self):
        """Transition to OPEN state"""
        with self._lock:
            self._state = self.OPEN
            self._last_failure_time = datetime.now()
            logger.warning(
                f"Circuit breaker '{self.name}' OPEN: "
                f"{self._failure_count} failures reached threshold {self.failure_threshold}"
            )
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        with self._lock:
            self._state = self.HALF_OPEN
            logger.info(f"Circuit breaker '{self.name}' HALF_OPEN: Testing recovery")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            logger.info(f"Circuit breaker '{self.name}' CLOSED: Service recovered")
    
    def _should_allow_request(self) -> bool:
        """Check if request should be allowed"""
        if self._state == self.CLOSED:
            return True
        
        if self._state == self.OPEN:
            # Check if recovery timeout has elapsed
            if self._last_failure_time and \
               datetime.now() - self._last_failure_time > timedelta(seconds=self.recovery_timeout):
                self._transition_to_half_open()
                return True
            return False
        
        # HALF_OPEN: Allow request to test recovery
        return True
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        if not self._should_allow_request():
            raise Exception(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service unavailable. Try again in {self.recovery_timeout}s"
            )
        
        try:
            result = func(*args, **kwargs)
            
            # Success - reset if in HALF_OPEN or reduce failure count
            if self._state == self.HALF_OPEN:
                self._transition_to_closed()
            elif self._failure_count > 0:
                with self._lock:
                    self._failure_count = max(0, self._failure_count - 1)
            
            return result
            
        except self.expected_exception as e:
            with self._lock:
                self._failure_count += 1
                
                if self._state == self.HALF_OPEN:
                    # Failed during recovery test
                    self._transition_to_open()
                elif self._failure_count >= self.failure_threshold:
                    self._transition_to_open()
            
            raise
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper


class RateLimiter:
    """
    Semaphore-based rate limiter for controlling concurrent requests
    Uses sliding window to track request rate
    """
    
    def __init__(
        self,
        max_concurrent: int,
        max_per_minute: Optional[int] = None,
        name: str = "RateLimiter",
    ):
        self.max_concurrent = max_concurrent
        self.max_per_minute = max_per_minute
        self.name = name
        
        self._semaphore = threading.Semaphore(max_concurrent)
        self._request_times: deque = deque()
        self._lock = threading.Lock()
        
        logger.info(
            f"Rate limiter '{name}' initialized: "
            f"max_concurrent={max_concurrent}, max_per_minute={max_per_minute}"
        )
    
    def _check_rate_limit(self):
        """Check if we're within rate limit"""
        if not self.max_per_minute:
            return True
        
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            
            # Remove old requests
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()
            
            if len(self._request_times) >= self.max_per_minute:
                logger.warning(
                    f"Rate limit exceeded for '{self.name}': "
                    f"{len(self._request_times)} requests in last minute"
                )
                return False
            
            self._request_times.append(now)
            return True
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire semaphore slot"""
        # Check rate limit first
        if not self._check_rate_limit():
            if not blocking:
                return False
            # Wait a bit and retry
            time.sleep(1)
            if not self._check_rate_limit():
                raise Exception(f"Rate limit exceeded for '{self.name}'")
        
        # Acquire semaphore
        return self._semaphore.acquire(blocking=blocking, timeout=timeout)
    
    def release(self):
        """Release semaphore slot"""
        self._semaphore.release()
    
    def __enter__(self):
        """Context manager entry"""
        if not self.acquire():
            raise Exception(f"Could not acquire rate limiter '{self.name}'")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with self:
                return func(*args, **kwargs)
        return wrapper
