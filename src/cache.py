"""
Caching layer for RAG system
Implements multi-level caching for embeddings, retrievals, and responses
"""

import hashlib
import json
import logging
from typing import Any, Optional, Dict, List
from collections import OrderedDict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        
        # Stats
        self._hits = 0
        self._misses = 0
        
        logger.info(f"LRU cache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if not self.ttl_seconds:
            return False
        
        timestamp = self._timestamps.get(key)
        if not timestamp:
            return True
        
        return datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check expiration
            if self._is_expired(key):
                self._remove(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        with self._lock:
            # Update existing
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
                self._timestamps[key] = datetime.now()
                return
            
            # Add new
            self._cache[key] = value
            self._timestamps[key] = datetime.now()
            
            # Evict oldest if needed
            if len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
    
    def _remove(self, key: str):
        """Remove key from cache"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.info("Cache cleared")
    
    def invalidate(self, key: str):
        """Invalidate specific key"""
        with self._lock:
            self._remove(key)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "utilization": f"{len(self._cache) / self.max_size * 100:.1f}%",
            }
    
    def __len__(self):
        return len(self._cache)


class EmbeddingCache:
    """Cache for query embeddings (dense and sparse)"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        logger.info("Embedding cache initialized")
    
    def _make_key(self, query: str, embedding_type: str) -> str:
        """Create cache key from query"""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{embedding_type}:{query_hash}"
    
    def get_dense(self, query: str) -> Optional[List[float]]:
        """Get cached dense embedding"""
        key = self._make_key(query, "dense")
        return self.cache.get(key)
    
    def put_dense(self, query: str, embedding: List[float]):
        """Cache dense embedding"""
        key = self._make_key(query, "dense")
        self.cache.put(key, embedding)
    
    def get_sparse(self, query: str) -> Optional[Any]:
        """Get cached sparse embedding"""
        key = self._make_key(query, "sparse")
        return self.cache.get(key)
    
    def put_sparse(self, query: str, embedding: Any):
        """Cache sparse embedding"""
        key = self._make_key(query, "sparse")
        self.cache.put(key, embedding)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.stats()


class RetrievalCache:
    """Cache for retrieval results"""
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 1800):
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        logger.info("Retrieval cache initialized")
    
    def _make_key(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> str:
        """Create cache key from query and parameters"""
        # Include query, filters, and top_k in key
        key_data = {
            "query": query,
            "filters": filters or {},
            "top_k": top_k,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"retrieval:{key_hash}"
    
    def get(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> Optional[List[Any]]:
        """Get cached retrieval results"""
        key = self._make_key(query, filters, top_k)
        result = self.cache.get(key)
        if result:
            logger.debug(f"Retrieval cache HIT for query: {query[:50]}...")
        return result
    
    def put(
        self,
        query: str,
        documents: List[Any],
        filters: Optional[Dict] = None,
        top_k: int = 5
    ):
        """Cache retrieval results"""
        key = self._make_key(query, filters, top_k)
        self.cache.put(key, documents)
        logger.debug(f"Cached retrieval for query: {query[:50]}...")
    
    def invalidate_all(self):
        """Invalidate all cached retrievals (e.g., after indexing new docs)"""
        self.cache.clear()
        logger.info("Retrieval cache invalidated")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.stats()


class ResponseCache:
    """Cache for complete LLM responses"""
    
    def __init__(self, max_size: int = 200, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        logger.info("Response cache initialized")
    
    def _make_key(self, query: str, context_hash: str) -> str:
        """Create cache key from query and context"""
        key_data = f"{query}:{context_hash}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"response:{key_hash}"
    
    def _hash_context(self, documents: List[Any]) -> str:
        """Create hash of document context"""
        # Hash document IDs/content to detect if same context
        doc_ids = [
            getattr(doc, 'id', '') or str(hash(getattr(doc, 'content', '')[:100]))
            for doc in documents
        ]
        context_str = '|'.join(sorted(doc_ids))
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]
    
    def get(self, query: str, documents: List[Any]) -> Optional[Any]:
        """Get cached response"""
        context_hash = self._hash_context(documents)
        key = self._make_key(query, context_hash)
        result = self.cache.get(key)
        if result:
            logger.debug(f"Response cache HIT for query: {query[:50]}...")
        return result
    
    def put(self, query: str, documents: List[Any], response: Any):
        """Cache response"""
        context_hash = self._hash_context(documents)
        key = self._make_key(query, context_hash)
        self.cache.put(key, response)
        logger.debug(f"Cached response for query: {query[:50]}...")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.stats()


class CacheManager:
    """Centralized cache manager for all caching layers"""
    
    def __init__(
        self,
        embedding_cache_size: int = 1000,
        retrieval_cache_size: int = 500,
        response_cache_size: int = 200,
        embedding_ttl: int = 3600,
        retrieval_ttl: int = 1800,
        response_ttl: int = 3600,
    ):
        self.embedding_cache = EmbeddingCache(
            max_size=embedding_cache_size,
            ttl_seconds=embedding_ttl
        )
        self.retrieval_cache = RetrievalCache(
            max_size=retrieval_cache_size,
            ttl_seconds=retrieval_ttl
        )
        self.response_cache = ResponseCache(
            max_size=response_cache_size,
            ttl_seconds=response_ttl
        )
        
        logger.info("Cache manager initialized with all layers")
    
    def clear_all(self):
        """Clear all caches"""
        self.embedding_cache.cache.clear()
        self.retrieval_cache.cache.clear()
        self.response_cache.cache.clear()
        logger.info("All caches cleared")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache layers"""
        return {
            "embedding_cache": self.embedding_cache.stats(),
            "retrieval_cache": self.retrieval_cache.stats(),
            "response_cache": self.response_cache.stats(),
        }
