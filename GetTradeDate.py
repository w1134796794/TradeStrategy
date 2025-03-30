from datetime import datetime
import akshare as ak
import pandas as pd
import logging
import bisect
from functools import lru_cache

logger = logging.getLogger(__name__)


class TradeCalendar:
    """
    获取最近的交易日：get_recent_trade_date
    获取前N个交易日,默认前一个交易日：get_previous_trade_date
    获取下一个交易日：get_next_trade_date
    """
    def __init__(self):
        self.trade_dates = self._load_trade_dates()
        self.sorted_dates = sorted(self.trade_dates) if self.trade_dates else []
        logger.debug(f"加载交易日历数据: {self.sorted_dates[-5:]}")  # 显示最近5个交易日

    def is_trade_date(self, date_str: str) -> bool:
        """
        验证日期是否为交易日
        :param date_str: 日期字符串（支持格式：YYYY-MM-DD 或 YYYYMMDD）
        :return: bool
        """
        try:
            # 统一格式处理
            if '-' in date_str:
                fmt_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
            else:
                fmt_date = date_str
            return fmt_date in self.sorted_dates
        except:
            return False

    @lru_cache(maxsize=1)
    def _load_trade_dates(self) -> list:
        """加载交易日数据并缓存"""
        try:
            today = datetime.now().strftime("%Y%m%d")
            df = ak.tool_trade_date_hist_sina()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            dates = [d.strftime("%Y%m%d") for d in df['trade_date']]
            logger.info(f"交易日历已更新（截止至{today}）")
            return dates
        except Exception as e:
            logger.error(f"接口请求失败，使用本地缓存示例数据: {str(e)}")
            return ['20250325', '20250326', '20250327', '20250328', '20250331']

    def clear_cache(self):
        """每日收盘后手动清理缓存"""
        self._load_trade_dates.cache_clear()

    def get_recent_trade_date(self):
        """获取最近的有效交易日（精确匹配优化版）"""
        if not self.sorted_dates:
            raise ValueError("无可用交易日数据")

        check_date = datetime.now()
        if isinstance(check_date, datetime):
            check_str = check_date.strftime("%Y%m%d")
        else:
            check_str = str(check_date)

        # 使用二分查找精确匹配
        index = bisect.bisect_left(self.sorted_dates, check_str)

        # 情况1：找到精确匹配
        if index < len(self.sorted_dates) and self.sorted_dates[index] == check_str:
            return check_str

        # 情况2：返回前一个有效交易日
        return self.sorted_dates[index - 1] if index > 0 else None

    def get_previous_trade_date(self, base_date=None, days=1):
        """
        获取基准日期前N个交易日
        :param base_date: 基准日期（默认当天）
        :param days: 向前追溯天数
        :return: 日期字符串（格式：YYYYMMDD）
        """
        if not self.sorted_dates:
            raise ValueError("无可用交易日数据")

        # 处理基准日期
        if base_date is None:
            base_str = datetime.now().strftime("%Y%m%d")
        elif isinstance(base_date, datetime):
            base_str = base_date.strftime("%Y%m%d")
        else:
            base_str = str(base_date)

        index = bisect.bisect_left(self.sorted_dates, base_date)
        if index > 0:
            return self.sorted_dates[index - days]
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

    def get_next_trade_date(self, base_date=None) -> None:
        """
        获取基准日期的下一个交易日
        :param base_date: 基准日期（支持格式：YYYY-MM-DD、YYYYMMDD 或 datetime 对象，默认当天）
        :return: 下一个交易日的字符串（格式：YYYYMMDD），如果已经是最后交易日返回None
        """
        if base_date is None:
            base_date = datetime.now().strftime("%Y%m%d")

        if not self.sorted_dates:
            raise ValueError("无可用交易日数据")

        # 处理基准日期格式
        if base_date is None:
            base_str = datetime.now().strftime("%Y%m%d")
        elif isinstance(base_date, datetime):
            base_str = base_date.strftime("%Y%m%d")
        else:
            base_str = str(base_date).replace("-", "")  # 统一为YYYYMMDD格式

        index = bisect.bisect_left(self.sorted_dates, base_date)
        if index < len(self.sorted_dates):
            next_date = self.sorted_dates[index]
            # 处理当天就是交易日的情况
            if next_date == base_date:
                if index + 1 < len(self.sorted_dates):
                    return self.sorted_dates[index + 1]
                else:
                    return None
            return next_date
        return None


# 测试示例
if __name__ == "__main__":

    calendar = TradeCalendar()


