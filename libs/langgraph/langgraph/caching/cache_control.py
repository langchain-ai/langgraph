from datetime import datetime, timedelta, timezone
from typing import Optional

class CacheControl:
    def __init__(self, cache_key: str, ttl: Optional[int] = None):
        """
        Args:
            cache_key (str): The unique identifier for caching.
            ttl (Optional[int]): Time-to-live in seconds, after which the cache is invalid.
        """
        self.cache_key = cache_key
        self.ttl = ttl

    def compute_time_bucket(self) -> str:
        """Calculate a precise time bucket string for the given TTL in seconds."""
        if not self.ttl:
            return 'no-ttl'
        now = datetime.now(timezone.utc)
        bucket_start = now - timedelta(seconds=now.timestamp() % self.ttl)
        return bucket_start.isoformat()