import pandas as pd
import logging
import psycopg2
import akshare as ak
from typing import Dict
from TradeDate import LocalTradeCalendar
from config import DB_CONFIG  # 假设数据库配置存储在config模块中

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据库连接工具 ====================
class DatabaseManager:
    """PostgreSQL数据库管理工具"""

    def __init__(self, db_config: dict):
        self.conn = psycopg2.connect(**db_config)

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """执行查询并返回DataFrame"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params or ())
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    return pd.DataFrame(data, columns=columns)
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}")
            raise


# ==================== 配置类 ====================
class MarketSentimentConfig:
    """市场情绪分析配置（数据库版）"""
    index_weights = {"sh000001": 0.4, "sz399001": 0.3, "sz399006": 0.3}
    score_weights = {'index': 0.4, 'breadth': 0.3, 'limit': 0.3, 'premium': 0.5}
    max_retries: int = 3
    new_stock_threshold: int = 5


# ==================== 核心分析类 ====================
class MarketSentimentAnalyzer:
    """市场情绪分析器（数据库版）"""

    def __init__(self, config: MarketSentimentConfig = None):
        self.config = config or MarketSentimentConfig()
        self.db_mgr = DatabaseManager(DB_CONFIG)
        self.calendar = LocalTradeCalendar()
        self.trade_date = self.calendar.get_recent_trade_date()

        # 初始化缓存
        self._index_data: Dict = {}
        self._market_breadth: Dict = {}
        self._limit_stats: Dict = {}

        # 指数名称映射
        self._index_name_map = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指"
        }

    # ==================== 数据获取方法 ====================
    def fetch_index_data(self) -> Dict:
        """从数据库获取指数数据"""
        index_data = {}
        query = """
            SELECT date, index_code, close 
            FROM index_data 
            WHERE index_code IN %s 
            ORDER BY date DESC 
            LIMIT 30
        """
        params = (tuple(self.config.index_weights.keys()),)

        try:
            df = self.db_mgr.execute_query(query, params)
            if df.empty:
                logger.warning("指数数据为空")
                return {}

            for code, weight in self.config.index_weights.items():
                code_df = df[df['index_code'] == code].sort_values('date')
                if len(code_df) < 2:
                    continue

                code_df['ma2'] = code_df['close'].rolling(2).mean()
                code_df['ma6'] = code_df['close'].rolling(6).mean()
                last_row = code_df.iloc[-1]
                prev_close = code_df.iloc[-2]['close']

                processed = {
                    'change_pct': (last_row['close'] - prev_close) / prev_close * 100,
                    'position': 'above' if last_row['close'] > last_row['ma2'] else 'below',
                    'trend': 'up' if last_row['ma2'] > last_row['ma6'] else 'down',
                    'weight': weight
                }
                index_data[code] = processed

            return index_data
        except Exception as e:
            logger.error(f"获取指数数据失败: {str(e)}")
            return {}

    def fetch_market_breadth(self) -> Dict:
        """从板块数据获取市场广度"""

        try:
            df = ak.stock_market_activity_legu()
            result = {
                'rise_num': int(df[df['item'] == '上涨']['value'].iloc[0]),
                'fall_num': int(df[df['item'] == '下跌']['value'].iloc[0]),
                'limit_up': int(df[df['item'] == '真实涨停']['value'].iloc[0]),
                'limit_down': int(df[df['item'] == '真实跌停']['value'].iloc[0])
            }

            # 保存缓存
            return result
        except Exception as e:
            logger.error(f"获取市场广度失败: {str(e)}")
            return {'rise_num': 0, 'fall_num': 0, 'limit_up': 0, 'limit_down': 0}

    def fetch_limit_stats(self) -> Dict:
        """从涨停池获取连板统计数据"""
        query = """
            SELECT 连板数, COUNT(*) AS count 
            FROM zt_pool_hist 
            WHERE 日期 = %s 
            GROUP BY 连板数
        """
        params = (self.trade_date,)

        try:
            df = self.db_mgr.execute_query(query, params)
            distribution = dict(zip(df['连板数'], df['count'])) if not df.empty else {}

            # 获取最高连板数
            max_limit = max(distribution.keys(), default=0)

            # 获取非连续涨停数据
            detail_query = """
                SELECT 代码, 名称, 涨停统计, 连板数 
                FROM zt_pool_hist 
                WHERE 日期 = %s
            """
            detail_df = self.db_mgr.execute_query(detail_query, params)

            non_consecutive = []

            for _, row in detail_df.iterrows():
                if '/' not in str(row['涨停统计']):
                    continue
                total, total_limit = map(int, row['涨停统计'].split('/'))
                non_consecutive.append({
                    'code': row['代码'],
                    'name': row['名称'],
                    'non_consecutive': total_limit - row['连板数']
                })

            return {
                'max_limit': max_limit,
                'distribution': distribution,
                'non_consecutive': non_consecutive
            }
        except Exception as e:
            logger.error(f"获取涨停数据失败: {str(e)}")
            return {'max_limit': 0, 'distribution': {}, 'non_consecutive': []}

    def fetch_market_volume_data(self, end_date: str, days: int = 5) -> pd.DataFrame:
        """
        从数据库获取市场成交量数据（优化版）

        参数：
            end_date: 截止日期（格式：YYYYMMDD）
            days: 需要获取的交易天数

        返回：
            DataFrame包含日期和总成交额（单位：亿元）
        """
        try:
            # 获取实际交易日期列表
            date_list = self.calendar.get_recent_trade_dates(end_date, days)
            if not date_list:
                logger.warning(f"无法获取{end_date}前{days}个交易日")
                return pd.DataFrame()

            # 构建SQL查询
            query = """
                SELECT 
                    date AS 日期,
                    SUM(成交额) / 1e8 AS 成交额   -- 直接转换为亿元
                FROM daily_stock_prices
                WHERE date IN %s
                GROUP BY date
                ORDER BY date DESC
            """

            # 执行查询
            volume_df = self.db_mgr.execute_query(query, (tuple(date_list),))

            if volume_df.empty:
                logger.warning(f"未找到{date_list}期间的成交数据")
                return pd.DataFrame()

            # 确保日期格式统一（处理数据库可能返回的不同日期格式）
            volume_df["日期"] = pd.to_datetime(volume_df["日期"]).dt.strftime("%Y%m%d")

            # 填充可能缺失的日期（设置为0）
            full_dates = pd.DataFrame({"日期": date_list})
            merged_df = full_dates.merge(volume_df, on="日期", how="left").fillna(0)

            # 按日期排序
            return merged_df.sort_values("日期").reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取量能数据失败: {str(e)}")
            return pd.DataFrame(columns=["日期", "成交额"])

    # ==================== 分析方法保持不变 ====================
    # （保持原有的calculate_total_score、analyze_premium_effect等方法逻辑，仅数据源改变）
    def calculate_volume_trend(self, end_date: str, days: int = 5) -> Dict:
        """基于数据库的量能趋势分析"""
        volume_df = self.fetch_market_volume_data(end_date, days)

        if volume_df.empty:
            return self._empty_volume_result()

        # 获取最后一天数据
        try:
            current_volume = volume_df.iloc[-1]["成交额"]
            avg_volume = volume_df.iloc[:-1]["成交额"].mean()
        except IndexError:
            return self._empty_volume_result()

        # 计算变化率（处理零值情况）
        if avg_volume == 0:
            change_ratio = 0.0
        else:
            change_ratio = (current_volume - avg_volume) / avg_volume

        # 生成结果（示例）
        return {
            "current_volume": round(current_volume, 2),
            "average_volume": round(avg_volume, 2),
            "change_ratio": round(change_ratio, 4),
            "status": self._get_volume_status(change_ratio)
        }

    def calculate_total_score(self) -> float:
        """优化版市场情绪评分（数据库版）"""
        try:
            total = 0.0
            end_date = self.trade_date

            # ==================== 量能分析维度 (35%) ====================
            # 获取近5日成交额数据
            date_list = self.calendar.get_recent_trade_dates(end_date, 5)
            query = """
                SELECT 日期 AS date, SUM(成交额) AS total_amount 
                FROM daily_stock_prices 
                WHERE 日期 IN %s 
                GROUP BY 日期
            """
            volume_df = self.db_mgr.execute_query(query, (tuple(date_list),))

            if volume_df.empty:
                logger.warning("成交额数据为空")
                return 0.0

            # 计算平均成交额和当日成交额
            current_volume = volume_df[volume_df['date'].astype(str) == end_date]['total_amount'].values

            current_volume = float(current_volume[0]) / 1e8 if len(current_volume) > 0 else 0  # 转换为亿元
            avg_volume = volume_df[volume_df['date'].astype(str) != end_date]['total_amount'].mean() / 1e8

            # 计算量能变化率
            change_ratio = (current_volume - avg_volume) / avg_volume if avg_volume != 0 else 0

            # 量能基础得分
            if change_ratio > 0.2:
                vol_base = 25
            elif change_ratio > 0.1:
                vol_base = 20
            elif change_ratio < -0.15:
                vol_base = 5
            elif change_ratio < -0.05:
                vol_base = 10
            else:
                vol_base = 15

            # 量能趋势得分
            trend_score = (current_volume - avg_volume) / 500 * 5 if current_volume > avg_volume else 0
            total += min((vol_base + trend_score), 35)

            # ==================== 市场热度维度 (50%) ====================
            breadth = self._market_breadth
            valid_stocks = breadth['rise_num'] + breadth['fall_num']

            # 上涨强度得分
            rise_strength = 0
            if valid_stocks > 0:
                rise_ratio = breadth['rise_num'] / valid_stocks
                rise_strength = min(rise_ratio * 50, 30)
                if breadth['rise_num'] > 3500:
                    rise_strength += 10

            # 涨停效应得分
            limit_score = breadth['limit_up'] * 0.3
            down_penalty = -breadth['limit_down']
            total += min((rise_strength + limit_score + down_penalty), 50)

            # ==================== 指数协同维度 (15%) ====================
            index_gain = 0.0
            for data in self._index_data.values():
                # 将 Decimal 转换为浮点数
                change_pct_float = float(data['change_pct'])
                weight_float = float(data['weight'])  # 确保 weight 也是浮点数
                index_gain += change_pct_float * weight_float

            # 后续计算
            index_base = min(max(index_gain * 5, 0), 10)
            volume_bonus = 5 if (index_gain > 0.5 and change_ratio > 0.1) else 0
            total += min(index_base + volume_bonus, 15)

            return min(max(round(total, 1), 0), 100)

        except Exception as e:
            logger.error(f"评分计算失败: {str(e)}")
            return 0.0

    def analyze_premium_effect(self, days: int = 5) -> Dict:
        """首板溢价效应分析（数据库版）"""
        try:
            profits = []
            total_samples = 0
            date_list = self.calendar.get_recent_trade_dates(self.trade_date, days)

            for date in date_list[:-2]:
                # 获取当日首板股票
                query = """
                    SELECT 代码, 最新价 
                    FROM zt_pool_hist 
                    WHERE 日期 = %s AND 连板数 = 1
                """
                zt_df = self.db_mgr.execute_query(query, (date,))

                if zt_df.empty:
                    continue

                zt_df['代码'] = zt_df['代码'].astype(str)

                # 获取下一个交易日
                next_date = self.calendar.get_next_trade_date(date)
                day_after_next = self.calendar.get_next_trade_date(next_date)

                # 批量获取价格数据
                codes = tuple(zt_df['代码'])
                price_query = """
                    SELECT 股票代码, 日期, 开盘 
                    FROM daily_stock_prices 
                    WHERE 股票代码 IN %s 
                        AND 日期 IN %s 
                    ORDER BY 日期
                """
                price_df = self.db_mgr.execute_query(
                    price_query,
                    (codes, (next_date, day_after_next))
                )

                # 转换为多索引DataFrame便于查询
                price_df = price_df.pivot(index='股票代码', columns='日期', values='开盘')
                price_df.index = price_df.index.astype(str)  # 索引转字符串
                price_df.columns = price_df.columns.astype(str)  # 列名转字符串
                price_df = price_df.fillna(0)

                for _, row in zt_df.iterrows():
                    code = row['代码']
                    try:
                        next_open = price_df.loc[code, next_date]
                        day_after_open = price_df.loc[code, day_after_next]
                    except KeyError:
                        continue

                    if next_open <= 0 or day_after_open <= 0:
                        continue

                    profit = (day_after_open / next_open - 1) * 100
                    profits.append(profit)
                    total_samples += 1

            # 结果统计
            if total_samples == 0:
                return self._empty_premium_result()

            profit_samples = [p for p in profits if p > 0]
            loss_samples = [p for p in profits if p < 0]

            return {
                'profit_probability': len(profit_samples) / total_samples * 100,
                'avg_profit': sum(profits) / total_samples,
                'avg_loss': sum(loss_samples) / len(loss_samples) if loss_samples else 0,
                'winning_profit_rate': sum(profit_samples) / len(profit_samples) if profit_samples else 0,
                'total_samples': total_samples,
                'loss_samples': len(loss_samples),
                'profit_samples': len(profit_samples)
            }

        except Exception as e:
            logger.error(f"溢价分析失败: {str(e)}")
            return self._empty_premium_result()

    def _empty_premium_result(self) -> Dict:
        return {
            'profit_probability': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'winning_profit_rate': 0.0,
            'total_samples': 0
        }

    def generate_report(self) -> Dict:
        """生成市场情绪报告"""
        # 数据采集
        self._index_data = self.fetch_index_data()
        self._market_breadth = self.fetch_market_breadth()
        self._limit_stats = self.fetch_limit_stats()

        # 计算得分
        total_score = self.calculate_total_score()
        # 计算高开和低开的溢价率
        premium_effect = self.analyze_premium_effect()

        return {
            'trade_date': self.trade_date,
            'total_score': total_score,
            'sentiment_level': self._get_sentiment_label(total_score),
            'index_data': self._format_index_data(),
            'breadth_data': self._market_breadth,
            'limit_stats': self._limit_stats,
            'premium_effect': premium_effect
        }

    def _format_index_data(self) -> Dict:
        """格式化指数数据为中文"""
        formatted = {}
        for code, data in self._index_data.items():
            formatted[code] = {
                '名称': self._index_name_map.get(code, "未知指数"),
                '当日涨跌幅': f"{data.get('change_pct', 0):.2f}%",
                '趋势': data.get('trend', 'N/A')
            }
        return formatted

    def _get_sentiment_label(self, score: float) -> str:
        """根据评分获取情绪级别"""
        if score >= 80:
            return "极度乐观"
        elif score >= 60:
            return "乐观"
        elif score >= 40:
            return "中性"
        elif score >= 20:
            return "谨慎"
        else:
            return "悲观"

    def _get_volume_status(self, ratio: float) -> str:
        """量能状态判断"""
        if ratio > 0.2:
            return "明显放量"
        elif ratio > 0.1:
            return "温和放量"
        elif ratio < -0.15:
            return "显著缩量"
        elif ratio < -0.05:
            return "轻微缩量"
        else:
            return "平量"

    def _empty_volume_result(self) -> Dict:
        return {
            "current_volume": 0.0,
            "average_volume": 0.0,
            "change_ratio": 0.0,
            "status": "无数据"
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    analyzer = MarketSentimentAnalyzer()
    report = analyzer.generate_report()

    print(f"【市场情绪报告】{report['trade_date']}")
    print(f"综合情绪分: {report['total_score']:.1f}-{report['sentiment_level']}")
    print("指数数据:")
    for code, data in report['index_data'].items():
        print(f"- {data['名称']}: {data['当日涨跌幅']} ({data['趋势']})")

    print(f"\n涨停统计: 最高{report['limit_stats']['max_limit']}连板")
    print(f"涨停: {report['breadth_data']['limit_up']}只，跌停：{report['breadth_data']['limit_down']}只")
    print(f"分布情况: {report['limit_stats']['distribution']}")

    print("\n溢价效应分析:")
    print(f"样本数量: {report['premium_effect']['total_samples']}")
    print(f"盈利概率: {report['premium_effect']['profit_probability']:.1f}%")
    print(f"平均收益: {report['premium_effect']['avg_profit']:.2f}%")
    print(f"亏损率: {report['premium_effect']['avg_loss']:.2f}% (基于 {report['premium_effect']['loss_samples']} 个样本)")
    print(f"盈利率: {report['premium_effect']['winning_profit_rate']:.2f}% (基于 {report['premium_effect']['profit_samples']} 个样本)")
