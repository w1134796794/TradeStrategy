import pandas as pd
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import sql
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LHBAnalyzer:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.autocommit = True

    def _execute_query(self, query, params=None) -> pd.DataFrame:
        """执行SQL查询"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    return pd.DataFrame(cursor.fetchall(), columns=columns)
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"查询执行失败: {str(e)}")
            return pd.DataFrame()

    def load_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """加载龙虎榜数据"""
        query = sql.SQL("""
            SELECT 
                m.股票代码, m.股票名称, m.数据日期, m.收盘价, m.涨跌幅,
                m.上榜后1日, m.上榜后2日, m.上榜后5日,
                d.交易营业部名称, d.买入金额, d.卖出金额 
            FROM lhb_main m
            JOIN lhb_detail d 
            ON m.股票代码 = d.股票代码 AND m.数据日期 = d.数据日期
            WHERE m.数据日期 BETWEEN %s AND %s
        """)
        return self._execute_query(query, (start_date, end_date))

    def _validate_columns(self, df: pd.DataFrame, required_cols: List[str]) -> bool:
        """验证数据列完整性"""
        missing = set(required_cols) - set(df.columns)
        if missing:
            logger.error(f"缺失必要字段: {missing}")
            return False
        return True

    def analyze_strategy(self, df: pd.DataFrame, min_trades: int = 5) -> Dict[str, pd.DataFrame]:
        """
        核心分析方法
        :param min_trades: 最小有效交易次数阈值
        :return: 包含多维度分析结果的字典
        """
        if not self._validate_columns(df, ['交易营业部名称', '上榜后1日', '上榜后2日', '上榜后5日']):
            return {}

        # 营业部多周期胜率分析
        branch_stats = self._calculate_win_rates(df, min_trades)

        # 近期活跃度分析（最近30天）
        active_data = self._analyze_recent_activity(df, days=30)

        return {
            'branch_win_rates': branch_stats,
            'active_branches': active_data['branches'],
            'hot_stocks': active_data['stocks']
        }

    def _calculate_win_rates(self, df: pd.DataFrame, min_trades: int) -> Dict[str, pd.DataFrame]:
        """计算多周期胜率"""
        periods = [1, 2, 5]
        results = {}

        for p in periods:
            col = f'上榜后{p}日'
            stats = df.groupby('交易营业部名称', observed=True).agg(
                total_trades=(col, 'size'),
                win_rate=(col, lambda x: (x > 0).mean())
            ).query(f'total_trades >= {min_trades}').sort_values('win_rate', ascending=False)

            results[f'{p}day'] = {
                'top': stats.head(10),
                'bottom': stats.tail(10)
            }
        return results

    def _analyze_recent_activity(self, df: pd.DataFrame, days: int = 30) -> Dict:
        """分析近期活跃度"""
        if not self._validate_columns(df, ['数据日期']):
            return {'branches': None, 'stocks': None}

        try:
            df['数据日期'] = pd.to_datetime(df['数据日期'])
            cutoff_date = df['数据日期'].max() - timedelta(days=days)
            recent_df = df[df['数据日期'] > cutoff_date]

            return {
                'branches': recent_df['交易营业部名称'].value_counts().head(10),
                'stocks': recent_df['股票名称'].value_counts().head(10)
            }
        except Exception as e:
            logger.error(f"活跃度分析异常: {str(e)}")
            return {'branches': None, 'stocks': None}

    def generate_report(self, analysis_data: Dict) -> str:
        """生成文本分析报告"""
        report = []

        # 胜率分析
        for period in ['1day', '2day', '5day']:
            data = analysis_data['branch_win_rates'].get(period, {})
            report.append(f"\n=== {period.upper()} 胜率分析 ===")
            report.append("\nTop 10营业部:")
            report.append(str(data.get('top', pd.DataFrame())))
            report.append("\nBottom 10营业部:")
            report.append(str(data.get('bottom', pd.DataFrame())))

        # 活跃度分析
        report.append("\n=== 近期活跃度分析 ===")
        report.append("\n活跃营业部Top10:")
        report.append(str(analysis_data['active_branches']))
        report.append("\n热门股票Top10:")
        report.append(str(analysis_data['hot_stocks']))

        return '\n'.join(map(str, report))


# 使用示例
if __name__ == "__main__":
    analyzer = LHBAnalyzer()

    # 数据加载（最近三个月）
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    raw_data = analyzer.load_data(start_date, end_date)

    if not raw_data.empty:
        analysis = analyzer.analyze_strategy(raw_data)
        report = analyzer.generate_report(analysis)
        print(report)
    else:
        print("未获取到有效数据")