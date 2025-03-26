from datetime import datetime
import akshare as ak
import pandas as pd

import bisect  # 添加bisect模块


class TradeCalendar:

    def __init__(self):
        self.trade_dates = self._load_trade_dates()
        # 将日期转换为排序后的列表（升序）
        self.sorted_dates = sorted(self.trade_dates) if self.trade_dates else []

    def is_market_hours(self):
        """判断当前是否在交易时段内（9:30-15:00）"""

        now = datetime.now().time()
        return (now >= datetime.strptime("09:00", "%H:%M").time() and now <= datetime.strptime("17:00", "%H:%M").time())

    def should_generate_next_day_plan(self):
        """判断是否需要生成次日计划"""
        now = datetime.now()
        current_time = now.time()
        # 条件1：交易时段外（15:00后）
        # 条件2：周末且周五收盘后
        return (
                (not self.is_market_hours()) or
                (now.weekday() >= 5)  # 周六/周日
        )

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
    def _load_trade_dates(self):
        """加载近3年的交易日历（返回排序列表）"""
        try:
            df = ak.tool_trade_date_hist_sina()
            # 转换日期格式并排序
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            return sorted([d.strftime("%Y%m%d") for d in df['trade_date']])
        except Exception as e:
            print(f"AKShare接口异常，使用本地缓存: {str(e)}")
            return self._load_local_calendar()

    def _load_local_calendar(self):
        return [
            '20250303', '20250304', '20250305', '20250306', '20250307'  # 新增测试数据
        ]

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

        # 查找基准日期的位置
        index = bisect.bisect_left(self.sorted_dates, base_str)

        # 处理基准日期非交易日的情况
        if index == 0:
            raise ValueError("基准日期早于所有交易日")
        if index >= len(self.sorted_dates) or self.sorted_dates[index] != base_str:
            index -= 1

        # 计算目标位置
        target_index = index - days
        if target_index < 0:
            raise ValueError(f"无法找到前{days}个交易日，最早日期为{self.sorted_dates[0]}")

        return self.sorted_dates[target_index]

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
        if not self.sorted_dates:
            raise ValueError("无可用交易日数据")

        # 处理基准日期格式
        if base_date is None:
            base_str = datetime.now().strftime("%Y%m%d")
        elif isinstance(base_date, datetime):
            base_str = base_date.strftime("%Y%m%d")
        else:
            base_str = str(base_date).replace("-", "")  # 统一为YYYYMMDD格式

        # 查找基准日期的位置
        index = bisect.bisect_left(self.sorted_dates, base_str)

        # 处理基准日期非交易日的情况
        if index < len(self.sorted_dates) and self.sorted_dates[index] != base_str:
            pass  # 保持index为插入位置
        else:
            index += 1  # 如果是交易日则找下一个

        # 返回下一个交易日
        try:
            return self.sorted_dates[index]
        except IndexError:
            return None  # 已经是最后一个交易日

# 测试示例
if __name__ == "__main__":

    calendar = TradeCalendar()

    # 测试案例1：正常情况
    print(calendar.get_recent_trade_date())

    print(calendar.should_generate_next_day_plan())

