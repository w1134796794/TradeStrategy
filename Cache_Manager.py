import pickle
from pathlib import Path
from datetime import datetime, timedelta, date
import hashlib
import functools
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def sort_iterable(obj):
    if isinstance(obj, list):
        # 对列表元素进行排序
        return sorted(sort_iterable(item) for item in obj)
    elif isinstance(obj, dict):
        # 对字典键值对进行排序
        return {k: sort_iterable(v) for k, v in sorted(obj.items())}
    return obj


class FileCache:
    def __init__(self, cache_dir="cache", ttl=3600):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, prefix=None, ttl=None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 对参数进行排序
                sorted_args = sort_iterable(args)
                sorted_kwargs = sort_iterable(kwargs)
                # 生成唯一缓存键（不包含日期）
                key = self._generate_key(func, sorted_args, sorted_kwargs, prefix)
                cache_file = self.cache_dir / f"{key}.pkl"

                # 检查缓存有效性
                if self._is_cache_valid(cache_file, ttl):
                    logger.debug(f"Loading cache: {cache_file}")
                    return self._load_cache(cache_file)

                # 执行原函数并缓存
                result = func(*args, **kwargs)
                self._save_cache(cache_file, result)
                return result

            return wrapper

        return decorator

    def _generate_key(self, func, args, kwargs, prefix):
        arg_hash = hashlib.md5(
            f"{args}{kwargs}".encode()
        ).hexdigest()[:8]
        # 格式：prefix_arg_hash
        return f"{prefix or func.__name__}_{arg_hash}"

    def _is_cache_valid(self, cache_file, ttl_override):
        """检查缓存是否有效（保留TTL逻辑）"""
        ttl = self.ttl if ttl_override is None else ttl_override
        if not cache_file.exists():
            return False
        file_mtime = cache_file.stat().st_mtime
        return (datetime.now().timestamp() - file_mtime) < ttl

    def _save_cache(self, cache_file, data):
        """使用pickle保存缓存数据"""
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"缓存保存失败: {e}")
            if cache_file.exists():
                cache_file.unlink()  # 删除不完整文件
            logger.debug(
                f"触发错误的数据预览:\n{data.head(2).to_string() if isinstance(data, pd.DataFrame) else repr(data)[:200]}")

    def _load_cache(self, cache_file):
        """使用pickle加载缓存数据"""
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None


# 全局缓存实例
default_cache = FileCache()