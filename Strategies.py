import pandas as pd
import numpy as np
import re
import logging

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