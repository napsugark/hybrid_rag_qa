"""
Langfuse integration for RAG query tracking and observability
Uses the modern @observe decorator pattern from langfuse
"""

import logging
from . import config

logger = logging.getLogger("HybridRAG")


def setup_langfuse():
    """
    Initialize and return Langfuse client.
    In Langfuse v3, Langfuse() constructor must be called to set up the
    OpenTelemetry tracer that @observe depends on. get_client() alone
    does NOT initialize the tracer.
    Returns None if Langfuse is disabled or unavailable.
    """
    if config.LANGFUSE_ENABLED:
        try:
            from langfuse import Langfuse
            
            # Langfuse() reads env vars and initializes the OTel tracer
            langfuse = Langfuse()
            logger.info("Langfuse tracking enabled (OTel tracer initialized)")
            logger.info(f"  - Host: {config.LANGFUSE_HOST}")
            return langfuse
            
        except ImportError:
            logger.warning("Langfuse package not installed. Run: poetry add langfuse")
        except Exception as e:
            logger.warning(f"Could not initialize Langfuse: {e}")
            logger.info("Continuing without Langfuse tracking")
    else:
        logger.info("Langfuse tracking disabled (no credentials in .env)")
    
    return None


def get_observe_decorator():
    """
    Get the @observe decorator if Langfuse is available.
    Returns a no-op decorator if Langfuse is disabled.
    """
    if config.LANGFUSE_ENABLED:
        try:
            from langfuse import observe
            logger.debug("Langfuse @observe decorator loaded")
            return observe
        except ImportError:
            logger.debug("Langfuse decorators not available")
    
    # Return no-op decorator if Langfuse not available
    def noop_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    
    return noop_decorator
