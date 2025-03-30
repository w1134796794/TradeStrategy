import akshare as ak
import pandas as pd
from typing import List
from GetTradeDate import TradeCalendar
from functools import lru_cache

from Cache_Manager import default_cache
pd.set_option("display.max_column", None)


class LHBProcessor:
    """
    修正版龙虎榜处理器（修复列名问题）
    关键修复：
    1. 调整营业部数据列名映射
    2. 添加缺失列处理逻辑
    """

    def __init__(self, max_retry: int = 3):
        self.max_retry = max_retry
        self.statistics_cache = {}
        self.calendar = TradeCalendar()

    @lru_cache(maxsize=3)  # 缓存最近3个周期的数据
    def _get_statistic_data(self, period: str = "近一月") -> pd.DataFrame:
        """获取统计周期内的全部股票上榜数据"""
        try:
            if period not in self.statistics_cache:
                df = ak.stock_lhb_stock_statistic_em(symbol=period)
                df['代码'] = df['代码'].astype(str).str.zfill(6)
                self.statistics_cache[period] = df
            return self.statistics_cache[period]
        except Exception as e:
            print(f"统计信息获取失败（{period}）: {e}")
            return pd.DataFrame()

    def clear_cache(self):
        """手动清理统计缓存"""
        self._get_statistic_data.cache_clear()

    def _get_basic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取当日龙虎榜基础数据"""
        try:
            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            return df[['代码', '名称', '上榜日', '解读', '涨跌幅', '上榜原因']]
        except Exception as e:
            print(f"基础数据获取失败 {start_date}-{end_date}: {e}")
            return pd.DataFrame()

    def _get_trade_details(self, code: str, date: str) -> pd.DataFrame:
        """获取营业部交易明细（修复列名问题）"""
        combined = []
        for flag in ["买入", "卖出"]:
            for _ in range(self.max_retry):
                try:
                    df = ak.stock_lhb_stock_detail_em(
                        symbol=code,
                        date=date,
                        flag=flag
                    )
                    if not df.empty:
                        # 统一列名格式（根据实际接口返回调整）
                        df = df.rename(columns={
                            '交易营业部名称': '营业部名称',  # 实际列名可能已经是这个
                            '买入金额': '买入金额',  # 根据实际返回字段调整
                            '买入金额-占总成交比例': '买入占比',
                            '卖出金额': '卖出金额',
                            '卖出金额-占总成交比例': '卖出占比'
                        })
                        df['代码'] = code
                        combined.append(df)
                    break
                except Exception as e:
                    print(f"获取{code} {flag}数据失败: {e}")
                    continue
        return pd.concat(combined) if combined else pd.DataFrame()

    def _enrich_statistics(self,
                           main_df: pd.DataFrame,
                           period: str = "近一月") -> pd.DataFrame:
        """合并统计指标（修复日期处理）"""
        stat_df = self._get_statistic_data(period)
        if stat_df.empty:
            return main_df

        # 转换日期格式
        stat_df['最近上榜日'] = pd.to_datetime(stat_df['最近上榜日'], format='%Y%m%d', errors='coerce')

        # 计算统计指标
        stat_agg = stat_df.groupby('代码').agg(
            累计上榜次数=('代码', 'count'),
            最近上榜日=('最近上榜日', 'max')
        ).reset_index()

        # 合并数据
        merged_df = main_df.merge(stat_agg, on='代码', how='left')

        # 计算近三日上榜
        if not merged_df.empty:
            latest_date = pd.to_datetime(merged_df['上榜日'], format='%Y%m%d').max()
            three_days_ago = latest_date - pd.DateOffset(days=3)
            merged_df['近三日上榜'] = merged_df.apply(
                lambda row: len(stat_df[
                                    (stat_df['代码'] == row['代码']) &
                                    (stat_df['最近上榜日'] >= three_days_ago)
                                    ]), axis=1
            )

        return merged_df.fillna({'累计上榜次数': 0, '近三日上榜': 0})

    def _process_trade_details(self, main_df: pd.DataFrame) -> pd.DataFrame:
        """处理营业部数据展开（优化版）"""
        # 展开嵌套数据时直接提取所需字段
        records = []
        for _, row in main_df.iterrows():
            # 基础信息
            base_info = {
                '代码': row['代码'],
                '名称': row['名称'],
                '上榜日': row['上榜日'],
                '解读': row['解读'],
                '涨跌幅': row['涨跌幅'],
                '上榜原因': row['上榜原因']
            }

            # 展开营业部明细
            for detail in row['营业部数据']:
                record = {
                    **base_info,
                    '营业部名称': detail.get('营业部名称', '未知席位'),
                    '买入金额': detail.get('买入金额', 0),
                    '买入占比': detail.get('买入金额-占总成交比例', 0),
                    '卖出金额': detail.get('卖出金额', 0),
                    '卖出占比': detail.get('卖出金额-占总成交比例', 0)
                }
                records.append(record)

        return pd.DataFrame(records)

    @default_cache(prefix="lhb_cache", ttl=86400)
    def get_enhanced_data(self,
                          dates: List[str],
                          statistic_period: str = "近一月") -> pd.DataFrame:
        """
        获取完整数据集（修复列缺失问题）
        """
        all_data = []

        # 获取基础数据
        for date in dates:
            base_df = self._get_basic_data(date, date)
            if base_df.empty:
                continue

            # 获取营业部明细
            for _, row in base_df.iterrows():
                detail_df = self._get_trade_details(row['代码'], date)
                if not detail_df.empty:
                    merged_row = row.to_dict()
                    merged_row['营业部数据'] = detail_df.to_dict('records')
                    all_data.append(merged_row)

        # 合并数据
        main_df = pd.DataFrame(all_data)
        if main_df.empty:
            return pd.DataFrame()

        final_df = self._process_trade_details(main_df)

        # 添加统计指标
        final_df = self._enrich_statistics(final_df, statistic_period)

        # 类型转换（处理可能缺失的列）
        numeric_cols = ['涨跌幅', '买入金额', '买入占比', '卖出金额', '卖出占比']
        for col in numeric_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            else:
                final_df[col] = 0.0  # 对缺失列填充默认值

        return final_df[
            ['代码', '名称', '上榜日', '解读', '涨跌幅', '上榜原因',
             '累计上榜次数', '最近上榜日', '近三日上榜',
             '营业部名称', '买入金额', '买入占比', '卖出金额', '卖出占比']
        ].dropna(subset=['营业部名称'])


# 使用示例
if __name__ == "__main__":
    processor = LHBProcessor()
    data = processor.get_enhanced_data(
        dates=['20250328'],
        statistic_period="近一月"
    )
    print(data.head(10))