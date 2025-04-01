from datetime import datetime
import pandas as pd
import logging
import bisect
from functools import lru_cache
import os

logger = logging.getLogger(__name__)


class LocalTradeCalendar:
    """
    本地交易日历管理（优先读取本地CSV，支持手动更新）
    包含：最近交易日/前后N交易日/区间交易日查询
    """
    CSV_PATH = './data/sse_trading_days_2025.csv'  # 固定本地路径
    CACHE_EXAMPLE = ['20250325', '20250326', '20250327', '20250328', '20250331']  # 最小可用缓存

    def __init__(self):
        self.trade_dates = self._load_local_dates()
        self.sorted_dates = sorted(self.trade_dates) if self.trade_dates else []
        self._log_initialization()

    @lru_cache(maxsize=1)
    def _load_local_dates(self) -> list:
        """优先读取本地CSV，失败时使用缓存示例"""
        try:
            if not os.path.exists(self.CSV_PATH):
                raise FileNotFoundError("本地交易日文件不存在")

            # 读取CSV并转换日期格式
            df = pd.read_csv(self.CSV_PATH, dtype={'trading_date': str})
            dates = df['trading_date'].tolist()

            # 校验数据完整性
            if not dates or len(dates[0]) != 8:
                raise ValueError("交易日数据格式异常")

            logger.info(f"成功加载本地交易日历（{len(dates)}条）")
            return dates

        except Exception as e:
            logger.warning(f"本地文件加载失败，使用缓存示例数据：{str(e)}")
            return self.CACHE_EXAMPLE  # 提供最小可用集合

    def _log_initialization(self):
        """初始化日志输出"""
        if self.sorted_dates:
            logger.debug(f"交易日历已加载!")
        else:
            logger.error("警告：无任何交易日数据，所有查询将返回异常")

    def clear_cache(self):
        """手动触发缓存更新（需配合文件更新）"""
        self._load_local_dates.cache_clear()
        self.trade_dates = self._load_local_dates()
        self.sorted_dates = sorted(self.trade_dates)

    # 以下为核心查询方法（保留原有逻辑，优化参数校验）
    def get_recent_trade_date(self):
        """获取最近交易日（含今日匹配）"""
        if not self.sorted_dates: return None
        today = datetime.now().strftime("%Y%m%d")
        idx = bisect.bisect_right(self.sorted_dates, today) - 1
        return self.sorted_dates[idx] if idx >= 0 else None

    def get_previous_trade_date(self, base_date=None, days=1):
        """获取前N个交易日（支持日期字符串/None）"""
        base = self._parse_date(base_date) or self.get_recent_trade_date()
        if not base: return None

        idx = bisect.bisect_left(self.sorted_dates, base)
        return self.sorted_dates[idx - days] if (idx - days) >= 0 else None

    def get_next_trade_date(self, base_date=None):
        """获取下一个交易日（支持日期字符串/None）"""
        base = self._parse_date(base_date) or datetime.now().strftime("%Y%m%d")
        idx = bisect.bisect_right(self.sorted_dates, base)
        return self.sorted_dates[idx] if idx < len(self.sorted_dates) else None

    # 新增辅助方法
    def _parse_date(self, date) -> str:
        """统一日期格式解析"""
        if isinstance(date, datetime):
            return date.strftime("%Y%m%d")
        if isinstance(date, str):
            return date.replace("-", "") if len(date) == 10 else date
        return None

    def get_trade_days(self, start_date: str, end_date: str) -> list:
        """
        获取指定日期区间内的交易日列表
        :param start_date: 开始日期 (格式: YYYYMMDD)
        :param end_date: 结束日期 (格式: YYYYMMDD)
        :return: 排序后的交易日列表（包含起止日期）
        """
        try:
            start_idx = self.trade_dates.index(start_date)
            end_idx = self.trade_dates.index(end_date)
            return self.trade_dates[start_idx:end_idx + 1]
        except ValueError:
            return []

    def get_trade_days_count(self, start_date: str, end_date: str) -> int:
        """
        计算两个日期之间的实际交易日数量（包含起止日）
        :param start_date: 开始日期 (格式: YYYYMMDD)
        :param end_date: 结束日期 (格式: YYYYMMDD)
        :return: 交易日数量（0表示无效查询）
        """
        # 参数有效性检查
        if (len(start_date) != 8 or len(end_date) != 8
                or start_date > end_date):
            return 0

        # 获取交易日列表
        trade_days = self.get_trade_days(start_date, end_date)
        return len(trade_days)


# 使用示例（保留原有测试逻辑）
if __name__ == "__main__":
    # 初始化日志
    logging.basicConfig(level=logging.DEBUG)

    # 初始化日历（自动读取本地文件）
    calendar = LocalTradeCalendar()

    # 测试查询
    current_date = '20250401'
    print(f"下一个交易日：{calendar.get_next_trade_date(current_date)}")
    print(f"前一个交易日：{calendar.get_previous_trade_date(current_date, 3)}")
    print(f"最近交易日：{calendar.get_recent_trade_date()}")


