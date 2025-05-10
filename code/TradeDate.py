from datetime import datetime
import pandas as pd
import logging
import bisect
import os
from typing import Tuple, List

logger = logging.getLogger(__name__)


class LocalTradeCalendar:
    """
    本地交易日历管理（优先读取本地CSV，支持手动更新）
    包含：最近交易日/前后N交易日/区间交易日查询
    """
    CSV_PATH = '../code/data/sse_trading_days_2025.csv'  # 固定本地路径
    CACHE_EXAMPLE = ['2025-03-25', '2025-03-26', '2025-03-27', '2025-03-28', '2025-03-31']  # 最小可用缓存

    def __init__(self):
        self.trade_dates = self._load_local_dates()
        self.sorted_dates = sorted(self.trade_dates) if self.trade_dates else []
        self._log_initialization()

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

            # 转换为 YYYY-MM-DD 格式
            formatted_dates = [self._format_date(date) for date in dates]
            # logger.info(f"成功加载本地交易日历（{len(formatted_dates)}条）")
            return formatted_dates

        except Exception as e:
            logger.warning(f"本地文件加载失败，使用缓存示例数据：{str(e)}")
            return self.CACHE_EXAMPLE  # 提供最小可用集合

    def _log_initialization(self):
        """初始化日志输出"""
        if self.sorted_dates:
            logger.debug(f"交易日历已加载!")
        else:
            logger.error("警告：无任何交易日数据，所有查询将返回异常")

    # 以下为核心查询方法（保留原有逻辑，优化参数校验）
    def get_recent_trade_date(self) -> str:
        """获取最近交易日（含今日匹配），格式为 YYYY-MM-DD"""
        if not self.sorted_dates:
            return None
        today = datetime.now().strftime("%Y-%m-%d")
        idx = bisect.bisect_right(self.sorted_dates, today) - 1
        return self.sorted_dates[idx] if idx >= 0 else None

    def get_previous_trade_date(self, base_date=None, days=1) -> str:
        """获取前N个交易日（支持日期字符串/None），格式为 YYYY-MM-DD"""
        base = self._parse_date(base_date) or self.get_recent_trade_date()
        if not base:
            return None

        idx = bisect.bisect_left(self.sorted_dates, base)
        return self.sorted_dates[idx - days] if (idx - days) >= 0 else None

    def get_next_trade_date(self, base_date=None) -> str:
        """获取下一个交易日（支持日期字符串/None），格式为 YYYY-MM-DD"""
        base = self._parse_date(base_date) or datetime.now().strftime("%Y-%m-%d")
        idx = bisect.bisect_right(self.sorted_dates, base)
        return self.sorted_dates[idx] if idx < len(self.sorted_dates) else None

    def get_recent_trade_dates(self, base_date: str, days: int, direction: str = "before") -> List[str]:
        """
        获取指定日期附近的交易日列表，格式为 YYYY-MM-DD

        Args:
            base_date: 基准日期(YYYY-MM-DD)
            days: 需要获取的天数
            direction: 方向，可选['before', 'after', 'both']

        Returns:
            按时间升序排列的日期列表
        """
        # 参数校验
        if not self._validate_date_format(base_date):
            raise ValueError("base_date格式应为YYYY-MM-DD")

        # 获取基准日期位置
        base_index = self._find_date_index(base_date)
        if base_index == -1:
            logger.warning(f"基准日期{base_date}非交易日，使用最近交易日替代")
            base_date = self.get_recent_trade_date()
            base_index = self._find_date_index(base_date)
            if base_index == -1:
                return []

        # 确定截取范围
        start_idx, end_idx = self._calculate_range(base_index, days, direction)

        # 处理边界情况
        start_idx = max(0, start_idx)
        end_idx = min(len(self.sorted_dates) - 1, end_idx)

        return self.sorted_dates[start_idx:end_idx + 1]

    def _calculate_range(self, base_index: int, days: int, direction: str) -> Tuple[int, int]:
        """计算日期范围索引"""
        if direction == "before":
            return base_index - days + 1, base_index
        elif direction == "after":
            return base_index, base_index + days - 1
        elif direction == "both":
            half = days // 2
            return base_index - half, base_index + half
        else:
            raise ValueError("direction参数应为['before','after','both']")

    def _find_date_index(self, date_str: str) -> int:
        """使用二分查找快速定位日期索引"""
        left, right = 0, len(self.sorted_dates) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.sorted_dates[mid] == date_str:
                return mid
            elif self.sorted_dates[mid] < date_str:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    def _validate_date_format(self, date_str: str) -> bool:
        """验证日期格式"""
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False

    # 新增辅助方法
    def _parse_date(self, date) -> str:
        """统一日期格式解析"""
        if isinstance(date, datetime):
            return date.strftime("%Y-%m-%d")
        if isinstance(date, str):
            # 兼容 YYYYMMDD 和 YYYY-MM-DD 格式
            if '-' in date:
                return date
            else:
                return self._format_date(date)
        return None

    def _format_date(self, date_str: str) -> str:
        """将日期格式从 YYYYMMDD 转换为 YYYY-MM-DD"""
        if isinstance(date_str, str) and len(date_str) == 8:
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return date_str

    def get_trade_days(self, start_date: str, end_date: str) -> list:
        """
        获取指定日期区间内的交易日列表，格式为 YYYY-MM-DD
        :param start_date: 开始日期 (格式: YYYY-MM-DD)
        :param end_date: 结束日期 (格式: YYYY-MM-DD)
        :return: 排序后的交易日列表（包含起止日期）
        """
        try:
            # 格式化输入日期
            start_date = self._parse_date(start_date)
            end_date = self._parse_date(end_date)

            # 查找起止日期的索引
            start_idx = bisect.bisect_left(self.sorted_dates, start_date)
            end_idx = bisect.bisect_right(self.sorted_dates, end_date) - 1

            # 返回格式化的日期列表
            return self.sorted_dates[start_idx:end_idx + 1]
        except ValueError:
            return []

    def get_trade_days_count(self, start_date: str, end_date: str) -> int:
        """
        计算两个日期之间的实际交易日数量（包含起止日）
        :param start_date: 开始日期 (格式: YYYY-MM-DD)
        :param end_date: 结束日期 (格式: YYYY-MM-DD)
        :return: 交易日数量（0表示无效查询）
        """
        # 参数有效性检查
        if not (self._validate_date_format(start_date) and self._validate_date_format(end_date)):
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
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"下一个交易日：{calendar.get_next_trade_date(current_date)}")
    print(f"前三个交易日：{calendar.get_previous_trade_date(current_date, 3)}")
    print(f"最近的交易日：{calendar.get_recent_trade_dates(current_date, 3)}")

    print(f"最近交易日：{calendar.get_recent_trade_date()}")