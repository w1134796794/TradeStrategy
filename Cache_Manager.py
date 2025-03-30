import json
from pathlib import Path
from functools import lru_cache
from datetime import datetime


class CacheManager:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def memory_cache(self, func):
        """通用内存缓存装饰器"""
        return lru_cache(maxsize=128)(func)

    def file_cache(self, key: str, ttl: int = 3600):
        """通用文件缓存装饰器（TTL: 秒）"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                cache_file = self.cache_dir / f"{key}.json"
                if cache_file.exists() and (datetime.now().timestamp() - cache_file.stat().st_mtime < ttl):
                    with open(cache_file, "r") as f:
                        return json.load(f)
                result = func(*args, **kwargs)
                with open(cache_file, "w") as f:
                    json.dump(result, f)
                return result
            return wrapper
        return decorator