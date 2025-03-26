from typing import Dict
import pandas as pd
import numpy as np


class ScoringStrategy:
    """股票综合评分策略"""

    def technical_score(cls, row: pd.Series) -> float:
        """技术面评分（40%）"""
        score = 0

        # 连板数得分（每个连板15分，上限30分）
        limit_score = min(row['连板数'] * 15, 30)

        # 封板时间得分（早盘涨停加分）
        if pd.notna(row['首次封板时间']):
            if int(row['首次封板时间'].split(':')[0]) < 10:
                limit_score += 10

        # 封单金额得分（亿元为单位的百分比）
        fund_score = min(np.log1p(row['封板资金'] / 1e8) * 5, 10)

        return limit_score + fund_score

    def capital_score(cls, row: pd.Series) -> float:
        """资金面评分（30%）"""
        score = 0

        # 龙虎榜得分
        lhb_score = min(row.get('lhb_score', 0) * 0.3, 15)

        # 机构参与得分
        institution_score = 5 if row.get('has_institution', False) else 0

        # 游资联动得分
        hot_spot_score = 10 if any('华鑫' in s for s in row.get('营业部列表', [])) else 0

        return lhb_score + institution_score + hot_spot_score

    def sector_score(cls, row: pd.Series) -> float:
        """板块评分（30%）"""
        if not isinstance(row['hot_sectors'], list):
            return 0

        # 取动量最高的两个板块
        sorted_sectors = sorted(row['hot_sectors'], key=lambda x: x[1], reverse=True)[:2]
        return sum(s[1] * 0.3 for s in sorted_sectors)

    def calculate_total_score(cls, row: pd.Series) -> float:
        """综合评分计算"""
        tech = cls.technical_score(row)
        capital = cls.capital_score(row)
        sector = cls.sector_score(row)
        total = tech + capital + sector
        return min(total, 100)


class PositionStrategy:
    """仓位管理策略"""

    _CONFIG = {
        'base_position': 20,  # 基础仓位百分比
        'market_adjust': {  # 市场情绪调整系数
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        },
        'limit_adjust': {  # 连板数调整
            3: 1.2,
            4: 1.5,
            5: 2.0
        }
    }

    def calculate_position(cls, row: pd.Series, market_status: Dict) -> float:
        """动态仓位计算"""
        # 基础仓位
        position = cls._CONFIG['base_position']

        # 市场情绪调整
        market_level = market_status.get('level', 'medium')
        position *= cls._CONFIG['market_adjust'][market_level]

        # 连板数调整
        for limit, ratio in cls._CONFIG['limit_adjust'].items():
            if row['连板数'] >= limit:
                position *= ratio

        # 机构参与调整
        if row.get('has_institution', False):
            position *= 1.5

        # 板块热度调整
        if any(s[1] > 80 for s in row.get('hot_sectors', [])):
            position *= 1.2

        return round(min(position, 40), 1)  # 单股仓位不超过40%


class RiskControlStrategy:
    """风险控制策略"""
    def check_market_risk(market_data: Dict) -> str:
        """市场风险等级评估"""
        if market_data['limit_down'] > 10:
            return 'high'
        if market_data['score'] < 40 and market_data['max_limit'] >= 5:
            return 'medium_high'
        if market_data['score'] < 50:
            return 'medium'
        return 'low'