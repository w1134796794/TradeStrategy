from pathlib import Path
from datetime import datetime, timedelta


def clean_cache(cache_dir="cache", days=7):
    cache_path = Path(cache_dir)
    cutoff_date = datetime.now() - timedelta(days=days)

    for f in cache_path.glob("*.json"):
        # 从文件名提取日期（格式：prefix_YYYYMMDD_...）
        date_str = f.stem.split("_")[-2]  # 根据实际键格式调整索引
        try:
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff_date:
                f.unlink()
                print(f"Deleted: {f}")
        except ValueError:
            # 处理不符合命名规范的文件
            pass


if __name__ == "__main__":
    clean_cache(days=3)