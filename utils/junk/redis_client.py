"""
Redis client utilities for caching
"""

import os
import logging
import redis.asyncio as redis
from utils.logger_config import get_logger

logger = get_logger(__name__)

_redis_client = None

async def init_redis():
    """Initialize Redis connection"""
    global _redis_client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    try:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        await _redis_client.ping()
        logger.info("✅ Redis connection initialized")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Caching will be disabled.")
        _redis_client = None

def get_redis_client():
    """Get Redis client instance"""
    return _redis_client

async def disconnect_redis():
    """Disconnect Redis client"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connection closed")
