import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketVolumeAnalyzer:
    MARKETS = {
        "sh": {"name": "沪市", "func": ak.stock_sh_a_spot_em},
        "sz": {"name": "深市", "func": ak.stock_sz_a_spot_em},
        "bj": {"name": "北交所", "func": ak.stock_bj_a_spot_em}
    }

    def __init__(self):
        self.trade_date = datetime.now().strftime("%Y%m%d")

    def get_total_amount(self, date: str = None) -> dict:
        """获取指定日期的总成交额(单位：亿元)并按市场、日期保存到本地"""
        if date is None:
            date = self.trade_date

        total = 0.0
        details = {}

        for market in self.MARKETS:
            try:
                df = self.MARKETS[market]["func"]()
                if not df.empty:
                    df = df.rename(columns={"成交额": "amount"})
                    df['amount'] = df['amount'].fillna(0)
                    df = df[["代码", "名称", "amount"]]
                    market_total = df["amount"].astype(float).sum() / 1e8
                    details[self.MARKETS[market]["name"]] = round(market_total, 2)
                    total += market_total

                # 按市场、日期保存到本地CSV
                self._save_to_csv(df, market, date)

            except Exception as e:
                logger.error(f"获取{self.MARKETS[market]['name']}数据失败: {str(e)}")

        # 保存汇总数据到本地CSV
        self._save_summary_to_csv(total, details, date)

        return {
            "total": round(total, 2),
            "detail": details,
            "date": date
        }

    def _save_to_csv(self, df: pd.DataFrame, market: str, date: str):
        """将市场数据按日期保存到本地CSV"""
        filename = f"data/market_volume/{market}/{date}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"数据已保存到 {filename}")

    def _save_summary_to_csv(self, total: float, details: dict, date: str):
        """将汇总数据保存到本地CSV"""
        summary = {
            "日期": [date],
            "总成交额(亿元)": round(total, 2),
            **{market: [amount] for market, amount in details.items()}
        }
        df = pd.DataFrame(summary)
        filename = f"data/market_volume/summary/{date}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"汇总数据已保存到 {filename}")

    def get_historical_average_amount(self, days: int = 5) -> float:
        """计算过去N个交易日的平均成交额"""
        total_amount = 0.0

        for i in range(1, days + 1):
            date = self._get_prev_trade_date(i)
            amount = self._get_historical_amount(date)
            total_amount += amount

        avg_amount = total_amount / days if days > 0 else 0
        return round(avg_amount, 2)

    def _get_prev_trade_date(self, days: int) -> str:
        """获取前N个交易日日期"""
        return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    def _get_historical_amount(self, date: str) -> float:
        """获取指定日期的成交额"""
        total = 0.0

        for market in self.MARKETS:
            try:
                df = self.MARKETS[market]["func"]()
                if not df.empty:
                    df = df.rename(columns={"成交额": "amount"})
                    market_total = df["amount"].astype(float).sum() / 1e8
                    total += market_total
            except Exception as e:
                logger.error(f"获取{self.MARKETS[market]['name']}历史数据失败: {str(e)}")

        return total

    def compare_with_historical_average(self, days: int = 5) -> dict:
        """比较当日成交量与过去N日平均成交量"""
        current_data = self.get_total_amount()
        avg_amount = self.get_historical_average_amount(days)

        change_ratio = (current_data["total"] - avg_amount) / avg_amount if avg_amount != 0 else 0
        change_ratio = round(change_ratio, 4)

        # 判断量能变化状态
        if change_ratio > 0.2:
            vol_status = "明显放量"
        elif change_ratio > 0.1:
            vol_status = "温和放量"
        elif change_ratio < -0.15:
            vol_status = "显著缩量"
        elif change_ratio < -0.05:
            vol_status = "轻微缩量"
        else:
            vol_status = "平量"

        return {
            "current": current_data["total"],
            "average": avg_amount,
            "change_ratio": change_ratio,
            "vol_status": vol_status,
            "days": days
        }


# 使用示例
if __name__ == "__main__":
    analyzer = MarketVolumeAnalyzer()

    # 获取当日成交额并保存
    print("当日成交额:", analyzer.get_total_amount())

    # 计算过去5日平均成交额
    avg_amount = analyzer.get_historical_average_amount(5)
    print(f"过去5日平均成交额: {avg_amount}亿")

    # 比较当日与过去5日平均成交额
    comparison = analyzer.compare_with_historical_average(5)
    print(f"""
    量能比较结果：
    当前总额：{comparison['current']}亿
    过去{comparison['days']}日平均：{comparison['average']}亿
    变化幅度：{comparison['change_ratio'] * 100:.2f}%
    量能状态：{comparison['vol_status']}
    """)
