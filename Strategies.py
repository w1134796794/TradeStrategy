import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import akshare as ak
from datetime import datetime
from GetHotSectors import SectorAnalyzer


logger = logging.getLogger(__name__)


class ScoringStrategy:
    """综合评分策略（完整修复版）"""

    @classmethod
    def filter_early_boards(cls, zt_df: pd.DataFrame) -> pd.DataFrame:
        """过滤首板二板"""
        try:
            zt_df['limit_count'] = zt_df['涨停统计'].apply(cls.parse_limit_count)
            return zt_df[zt_df['limit_count'].isin([1, 2])]
        except Exception as e:
            logger.error(f"过滤失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def parse_limit_count(zt_stat: str) -> int:
        """解析连板数"""
        try:
            return int(re.search(r'(\d+)连板', zt_stat).group(1)) if zt_stat else 1
        except:
            return 1

    @staticmethod
    def technical_score(row: pd.Series) -> float:
        """技术面评分"""
        try:
            limit_count = int(row.get('limit_count', 1))
        except (TypeError, ValueError):
            limit_count = 1
            logger.warning(f"无效的连板数: {row.get('limit_count')}, 已重置为1")

        try:
            seal_amount = float(row.get('seal_amount', 0))
        except (TypeError, ValueError):
            seal_amount = 0.0
            logger.warning(f"无效的封单金额: {row.get('seal_amount')}, 已重置为0")

        return min(limit_count * 15, 30) + min(np.log1p(seal_amount / 1e8) * 5, 10)

    @staticmethod
    def capital_score(row: pd.Series) -> float:
        """资金面评分"""
        lhb_score = float(row.get('lhb_score', 0))
        return min(lhb_score * 0.3, 15) + (5 if row.get('has_institution') else 0)

    @staticmethod
    def calculate_total_score(row: pd.Series) -> float:
        """综合评分（类型安全版）"""
        # 强制类型转换
        try:
            limit_count = int(row.get('limit_count', 1))
            seal_amount = float(row.get('seal_amount', 0))
            lhb_score = float(row.get('lhb_score', 0))
        except (ValueError, TypeError) as e:
            logger.error(f"数值转换失败: {e}")
            return 0.0

        # 计算逻辑
        tech_score = min(limit_count * 15, 30) + min(np.log1p(seal_amount / 1e8) * 5, 10)
        capital_score = min(lhb_score * 0.3, 15)

        return min(tech_score + capital_score, 100)


class PositionStrategy:
    """仓位策略（修复版）"""

    _CONFIG = {
        'base_position': 20,
        'market_adjust': {'high': 1.5, 'medium': 1.0, 'low': 0.5}
    }

    @classmethod
    def calculate_position(cls, row: pd.Series, market_status: dict) -> float:
        """动态仓位计算"""
        position = cls._CONFIG['base_position']
        position *= cls._CONFIG['market_adjust'].get(market_status['level'], 1.0)
        return min(position, 40)


class RiskControlStrategy:
    """风控策略（修复版）"""

    @staticmethod
    def check_market_risk(market_data: dict) -> str:
        if market_data['limit_down'] > 10:
            return 'high'
        return 'low' if market_data['score'] >= 50 else 'medium'


class DragonHeadStrategy:
    """龙头股评估策略（多维量化版）"""

    def __init__(self, config=None):
        self.config = config or {
            'days': 3,  # 近期涨幅计算天数
            'index_code': 'sh000001',  # 大盘指数代码
            'net_inflow_threshold': 5e7,  # 大单净流入阈值(万元)
            'early_limit_hour': 10,  # 早盘快速涨停时间(小时)
            'weights': {  # 评分权重
                'recent_gain': 0.3,
                'resist_market': 0.3,
                'big_order': 0.2,
                'fast_limit': 0.2
            }
        }
        self.sector_analyzer = SectorAnalyzer()

    def evaluate_sector(self, sector_stocks: List[str], index_data: pd.DataFrame) -> pd.DataFrame:
        """
        评估板块内个股的龙头潜力
        :param sector_stocks: 板块成分股代码列表
        :param index_data: 大盘指数历史数据(需包含日期、收盘价)
        :return: 带评分的DataFrame
        """
        # 获取个股多维数据
        stock_data = self._collect_stock_metrics(sector_stocks)

        # 计算大盘同期表现
        market_change = self._calculate_market_change(index_data)

        # 多维评分
        scores = []
        for code, metrics in stock_data.items():
            score = self._calculate_score(metrics, market_change)
            scores.append({**metrics, **score})

        df = pd.DataFrame(scores)
        if 'total_score' not in df.columns:
            logger.error("评分失败：未生成 total_score 列，可能原因：")
            logger.error("1. _calculate_score 未返回 total_score")
            logger.error(f"2. 数据样例：{df.head(2) if not df.empty else '空数据集'}")
            return pd.DataFrame()

        return df.sort_values('total_score', ascending=False)

    def _collect_stock_metrics(self, codes: List[str]) -> Dict:
        """收集个股多维指标"""
        metrics = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._get_single_stock_data, code): code for code in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    data = future.result()
                    if data: metrics[code] = data
                except Exception as e:
                    logger.error(f"股票{code}数据获取失败: {str(e)}")
        return metrics

    def _get_single_stock_data(self, code: str) -> Dict:
        """获取单只股票多维数据"""
        # 获取历史涨幅
        start_date = self.sector_analyzer._get_start_date(self.config['days'])
        hist_df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                     adjust="hfq",
                                     start_date=start_date)
        if len(hist_df) < self.config['days']:
            return None

        # 近期涨幅计算
        recent_gain = (hist_df['收盘'].iloc[-1] / hist_df['收盘'].iloc[-self.config['days']] - 1) * 100

        # 当日资金流
        spot_data = ak.stock_individual_fund_flow(code)
        if spot_data.empty:
            return None

        # 大单净流入（单位：万）
        big_order_net = spot_data[spot_data['日期'] == spot_data['日期'].max()]['大单净流入'].values
        big_order_net = big_order_net[0] if big_order_net else 0  # Default to 0 if no value is found

        # 涨停板数据
        zt_data = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
        zt_info = zt_data[zt_data['代码'] == code]
        first_limit_time = pd.to_datetime(zt_info['首次封板时间'].iloc[0]) if not zt_info.empty else None

        return {
            'code': code,
            'name': ak.stock_zh_a_spot_em(symbol=code)['名称'].iloc[0],
            'recent_gain': recent_gain,
            'big_order_net': big_order_net,
            'first_limit_time': first_limit_time,
            'close': hist_df['收盘'].iloc[-1]
        }

    def _calculate_market_change(self, index_data: pd.DataFrame) -> Dict:
        """计算大盘同期波动"""
        return {
            'market_change': (index_data['close'].iloc[-1] / index_data['close'].iloc[-self.config['days']] - 1) * 100,
            'daily_changes': index_data['close'].pct_change().tail(self.config['days']).values
        }

    def _calculate_score(self, metrics: Dict, market_data: Dict) -> Dict:
        """多维评分逻辑"""
        scores = {}

        # 1. 近期涨幅得分
        scores['gain_score'] = self._normalize(metrics['recent_gain'], 20, 50) * 100

        # 2. 逆势能力得分
        resist_score = 0
        for daily_change in market_data['daily_changes']:
            if daily_change < 0 and metrics['close'] > metrics['close'].shift(1):
                resist_score += 20  # 当日大盘跌但个股涨
        scores['resist_score'] = min(resist_score, 100)

        # 3. 大单净流入得分
        scores['big_order_score'] = 100 if metrics['big_order_net'] > self.config['net_inflow_threshold'] else 0

        # 4. 快速封板得分
        if metrics['first_limit_time'] and metrics['first_limit_time'].hour < self.config['early_limit_hour']:
            scores['fast_limit_score'] = 100
        else:
            scores['fast_limit_score'] = 0

        # 加权总分
        total = sum(
            scores[k] * self.config['weights'][k.split('_')[0]]
            for k in scores.keys()  # 明确遍历字典键
        )

        print(total)
        print('11111')
        return {
            **scores,
            'total_score': total
        }

    @staticmethod
    def _normalize(value, min_val, max_val):
        """归一化到0-1范围"""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))


if __name__ == "__main__":
    strategy = DragonHeadStrategy(config={
        'days': 3,
        'net_inflow_threshold': 8e7,  # 提高大单阈值
        'early_limit_hour': 10
    })

    # 获取半导体板块成分股
    sector_stocks = ['688981', '002371', '603501', '600703']

    # 获取上证指数数据
    index_data = ak.stock_zh_index_daily(symbol="sh000001")

    # 评估龙头潜力
    result = strategy.evaluate_sector(sector_stocks, index_data)
    print(result[['code', 'name', 'total_score']].head(5))