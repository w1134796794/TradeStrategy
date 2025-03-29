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

    _CONFIG = {
        # 权重配置（总和建议为1）
        "weights": {
            "board_level": 0.3,  # 连板层级权重（首板/二板）
            "sector_heat": 0.3,  # 板块热度权重
            "lhb_quality": 0.2,  # 龙虎榜席位质量权重
            "seal_amount": 0.2,  # 封板金额权重
        },
        # 龙虎榜知名席位名单（可配置）
        "famous_offices": ["赵老哥", "章盟主", "方新侠", "机构专用"],
        # 板块热度阈值（高于此值视为热门）
        "sector_heat_threshold": 80,
    }

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

    @classmethod
    def calculate_total_score(cls, row: pd.Series, sector_heat_map: Dict[str, float]) -> float:
        """
        计算股票综合得分（0-100分）
        :param row: 单只股票数据，需包含字段 ['limit_count', 'seal_amount', 'lhb_offices']
        :param sector_heat_map: 板块热度字典 {板块名: 热度分}
        :return: 综合得分
        """
        try:
            # ------------------------------
            # 1. 连板层级得分（0-30分）
            # ------------------------------
            # 首板=20分，二板=30分，其他=0分
            board_score = 20 if row['limit_count'] == 1 else 30 if row['limit_count'] == 2 else 0

            # ------------------------------
            # 2. 板块热度得分（0-30分）
            # ------------------------------
            sector_score = 0
            for sector in row.get('hot_sectors', []):
                sector_name = sector[0]  # 假设格式为 (板块名, 得分)
                heat = sector_heat_map.get(sector_name, 0)
                if heat >= cls._CONFIG["sector_heat_threshold"]:
                    sector_score += min(heat, 100)  # 单板块最多100分
            sector_score = min(sector_score, 30)  # 板块总分不超过30分

            # ------------------------------
            # 3. 龙虎榜席位质量得分（0-20分）
            # ------------------------------
            lhb_score = 0
            for office in row.get('lhb_offices', []):
                if office in cls._CONFIG["famous_offices"]:
                    lhb_score += 5  # 每个知名席位加5分
            lhb_score = min(lhb_score, 20)  # 总分不超过20分

            # ------------------------------
            # 4. 封板金额得分（0-20分）
            # ------------------------------
            # 封板金额单位：亿元，得分=金额*10（如2亿→20分）
            amount_score = min(row['seal_amount'] / 1e8 * 10, 20)

            # ------------------------------
            # 综合加权得分
            # ------------------------------
            total_score = (
                board_score * cls._CONFIG["weights"]["board_level"] +
                sector_score * cls._CONFIG["weights"]["sector_heat"] +
                lhb_score * cls._CONFIG["weights"]["lhb_quality"] +
                amount_score * cls._CONFIG["weights"]["seal_amount"]
            )
            return min(total_score, 100)  # 总分不超过100

        except Exception as e:
            logger.error(f"评分计算失败: {str(e)}")
            return 0


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


class FilterStrategy:
    """候选股过滤策略"""

    @staticmethod
    def filter_early_boards(zt_df: pd.DataFrame) -> pd.DataFrame:
        """
        过滤首板/二板股票
        :param zt_df: 涨停板数据，需包含字段 'limit_count'
        :return: 过滤后的DataFrame
        """
        return zt_df[zt_df['limit_count'].isin([1, 2])].copy()