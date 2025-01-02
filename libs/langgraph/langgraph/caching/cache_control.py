from datetime import datetime, timedelta, timezone
from hashlib import sha1
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
    
    def generate_task_id(self, input_data: str) -> str:
        """
        Generate a cache-enabled task ID based on cache key, stringified input data, and TTL bucket.

        Args:
            input_data (str): The stringified input data to be hashed for uniqueness.

        Returns:
            str: A unique task ID that includes the cache key, input hash, and TTL bucket.
        """
        # Hash the input data to create a unique identifier
        input_hash = sha1(input_data.encode()).hexdigest()
        
        # Compute the TTL bucket
        ttl_bucket = self.compute_time_bucket()
        
        # Combine the cache key, input hash, and TTL bucket to form the task ID
        return f"{self.cache_key}:{input_hash}:{ttl_bucket}"
