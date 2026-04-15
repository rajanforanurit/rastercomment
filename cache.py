# comment-service/cache.py
from functools import lru_cache
import threading

class CommentCache:
    def __init__(self, maxsize=1000):
        self.cache = {}
        self.lock = threading.Lock()
        self.maxsize = maxsize
    
    def get(self, key):
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.maxsize:
                # Simple eviction
                first_key = next(iter(self.cache))
                self.cache.pop(first_key)
            self.cache[key] = value

comment_cache = CommentCache()
