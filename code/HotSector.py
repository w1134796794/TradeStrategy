import psycopg2
from psycopg2 import sql
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from TradeDate import LocalTradeCalendar
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.set_option("display.max_columns", None)


class SectorConfig:
    """板块分析配置参数"""

    # 分析参数
    DAYS = 3  # 分析涨幅周期
    RESIST_DAYS = 5  # 逆势周期
    DAYS20 = 20
    HIST_DAYS = 25  # 历史周期
    TOP_N_PER_TYPE = 6  # 每类板块取前N名
    INDEX_CODE = 'sh000001'  # 基准指数代码

    # 权重配置
    WEIGHTS_CONFIG = {
        'breakthrough': {  # 突破形态得分
            'weight': 10,
            'calc': lambda x: min(x * 5, 10)  # 每突破一次关键位得5分
        },
        'funds_inflow': {  # 资金流入得分
            'weight': 15,
            'calc': lambda x: (
                # 分段非线性评分
                min(
                    max(0, np.log1p(abs(x)/5) * 8) if x > 0  # 正向流入
                    else max(-5, -np.log1p(abs(x)/2) * 3),   # 负向流出惩罚
                    15
                )
            )
        },
        'resist_days': {  # 逆势上涨
            'weight': 10,
            'calc': lambda x: min(x * 2, 10)  # 每天4分
        },
        'float_mv': {  # 流通市值
            'weight': 10,
            'calc': lambda x: 10 * (1 - np.log(x + 1) / np.log(2e10))  # 非线性映射
        },
        'consecutive': {  # 连板数
            'weight': 10,
            'calc': lambda x: min(x * 2, 10)
        },
        'volume_ratio_score': {
            'weight': 20,
            'calc': lambda x: max(0, 20 * (1 - (x - 2)**2 / 2.25))
        },
        'recent_gain': {  # 新增近期涨幅评分项
            'weight': 25,
            'calc': lambda x: (
                # S型曲线映射涨幅
                25 / (1 + np.exp(-0.3*(x - 10)))  # 10%涨幅时得12.5分，20%得20分，30%得23.6分
            )
        }
    }


class DatabaseManager:
    """数据库管理类"""

    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.conn.autocommit = True

    def execute_query(self, query, params=None):
        """执行查询并返回DataFrame"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    return pd.DataFrame(data, columns=columns)
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}")
            return pd.DataFrame()


db_manager = DatabaseManager()


class SectorAnalyzerDB:
    """使用数据库的板块分析类"""

    def __init__(self, trade_date: str = None):
        self.calendar = LocalTradeCalendar()
        self.trade_date = trade_date or self._fetch_recent_trade_date()

    # --------------------------
    # 数据库访问方法
    # --------------------------

    @staticmethod
    def fetch_sector_list(sector_type: str) -> pd.DataFrame:
        """从数据库获取板块列表"""
        query = sql.SQL("""
            SELECT * 
            FROM sector_list 
            WHERE 板块类型 = %s AND 板块名称 NOT LIKE '%%昨日%%'
        """)
        df = db_manager.execute_query(query, (sector_type,))
        return df

    @staticmethod
    def fetch_sector_history(sector: str, sector_type: str, days: int) -> pd.DataFrame:
        """从数据库获取板块历史数据"""
        query = sql.SQL("""
            SELECT * FROM sector_hist 
            WHERE 板块名称 = %s AND 板块类型 = %s 
            ORDER BY 日期 DESC LIMIT %s
        """)
        df = db_manager.execute_query(query, (sector, sector_type, days))
        if not df.empty:
            df['日期'] = pd.to_datetime(df['日期'])
            df.sort_values('日期', inplace=True)
        return df

    def fetch_zt_pool(self) -> pd.DataFrame:
        """从数据库获取涨停池数据"""
        query = sql.SQL("""
            SELECT * FROM zt_pool_hist 
            WHERE 日期 = %s
        """)
        df = db_manager.execute_query(query, (self.trade_date,))
        if not df.empty:
            df['代码'] = df['代码'].astype(str).str.zfill(6)
        return df

    @staticmethod
    def fetch_sector_components(sector: str, sector_type: str) -> List[str]:
        """从数据库获取板块成分股"""
        query = sql.SQL("""
            SELECT 代码 FROM sector_components 
            WHERE 板块名称 = %s AND 板块类型 = %s 
            AND 代码 NOT LIKE '900%%' AND 代码 NOT LIKE '20%%'
        """)
        df = db_manager.execute_query(query, (sector, sector_type))
        return df['代码'].astype(str).str.zfill(6).tolist()

    @staticmethod
    def fetch_sector_stock_hist(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从数据库获取个股历史数据"""
        query = sql.SQL("""
            SELECT * FROM daily_stock_prices 
            WHERE 股票代码 = %s 
            AND 日期 BETWEEN %s AND %s 
            ORDER BY 日期
        """)
        df = db_manager.execute_query(query, (symbol, start_date, end_date))
        if not df.empty:
            df.rename(columns={'股票代码': '代码'}, inplace=True)
            df['日期'] = pd.to_datetime(df['日期'])
        return df

    @staticmethod
    def fetch_index_data() -> pd.DataFrame:
        """从数据库获取指数数据"""
        query = sql.SQL("""
            SELECT date, close FROM index_data 
            WHERE index_code = %s 
            ORDER BY date DESC LIMIT 10
        """)
        df = db_manager.execute_query(query, (SectorConfig.INDEX_CODE,))
        if not df.empty:
            # 转换为日期类型并排序
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=True)  # 关键步骤：转为时间正序

            # 计算正确的时间顺序涨跌幅
            df['pct_change'] = df['close'].pct_change() * 100

            # 恢复时间倒序返回（保持与原始数据顺序一致）
            return df.sort_values('date', ascending=False)[['date', 'pct_change']]

        return pd.DataFrame()

    @staticmethod
    def fetch_float_market_value(code: str) -> float:
        """从数据库获取流通市值"""
        # 尝试从多个数据源获取
        query = sql.SQL("""
            SELECT 流通市值 FROM market_volume 
            WHERE 代码 = %s 
            ORDER BY 日期 DESC LIMIT 1
        """)
        df = db_manager.execute_query(query, (code,))
        if df.empty:
            query = sql.SQL("SELECT 流通市值 FROM zt_pool_hist WHERE 代码 = %s ORDER BY 日期 DESC LIMIT 1")
            df = db_manager.execute_query(query, (code,))

        if not df.empty:
            return float(df['流通市值'].iloc[0]) / 1e8
        return 2e10

    @staticmethod
    def fetch_fund_inflow(code: str) -> float:
        try:
            # 格式代码为6位字符串（如果数据库存储的是数字）
            formatted_code = str(code).zfill(6)

            # 查询最新一条记录
            query = sql.SQL("""
                SELECT * 
                FROM fund_flow 
                WHERE 代码 = %s 
                ORDER BY 日期 DESC 
                LIMIT 1
            """)
            df = db_manager.execute_query(query, (formatted_code,))

            if df.empty:
                logger.warning(f"无资金流数据: {formatted_code}")
                return 0.0

            # 获取关键字段
            main_inflow_pct = df['今日主力净流入净占比'].iloc[0]  # 百分比值
            super_inflow = df['今日超大单净流入净额'].iloc[0]  # 单位：元
            close_price = df['最新价'].iloc[0]  # 单位：元/股

            # 异常值过滤
            if abs(main_inflow_pct) > 30:
                if close_price <= 0:
                    return 0.0
                # 计算单位需根据业务需求调整
                # 假设 super_inflow 是元，转换为亿元除以股价
                return (super_inflow / 1e8) / close_price

            return main_inflow_pct

        except Exception as e:
            logger.error(f"资金流入数据获取失败 {code}: {str(e)}", exc_info=True)
            return 0.0

    # --------------------------
    # 分析计算方法
    # --------------------------
    # ---------热门板块----------
    @staticmethod
    def calculate_trend_score(hist_data: pd.DataFrame, n_days: int) -> float:
        """
        计算板块趋势得分（基于N日行情）
        优化后评分规则：
        - 价格动量（65%）：N日累计涨跌幅
        - 量能趋势（35%）：(当前成交量 - N日均量)/N日均量 * 价格方向系数
        """
        try:
            # 数据截取逻辑
            recent_data = hist_data.iloc[-n_days - 1:] if len(hist_data) > n_days else hist_data

            # 核心指标计算
            price_change = recent_data['涨跌幅'].sum()
            direction = 1 if price_change >= 0 else -1  # 价格方向系数

            # 量能趋势计算
            mean_volume = recent_data['成交量'].mean()
            current_volume = hist_data.iloc[-1]['成交量'] if len(hist_data) > 0 else 0
            volume_ratio = (
                ((current_volume - mean_volume) / mean_volume * direction)
                if mean_volume > 0
                else 0
            )

            # 加权得分计算
            return round(
                price_change * 0.7 +  # 价格动量权重
                volume_ratio * 100 * 0.3,  # 量能趋势权重（放大100倍处理百分比）
                2
            )
        except Exception as e:
            logger.error(f"趋势评分计算失败: {str(e)}", exc_info=True)
            return 0.0

    def analyze_single_sector(self, sector: str, sector_type: str, days: int) -> Optional[Tuple]:
        """单个板块分析流水线"""
        try:
            hist = self.fetch_sector_history(sector, sector_type, days)
            if hist.empty:
                return None

            score = self.calculate_trend_score(hist, days)
            return sector, sector_type, score
        except Exception as e:
            logger.error(f"板块分析异常 {sector}: {str(e)}")
            return None

    def analyze_sector_data(self, sector_type: str, days: int, top_n: int) -> List[Tuple]:
        """处理板块数据"""
        sectors = self.fetch_sector_list(sector_type)
        if sectors.empty:
            logger.warning(f"{sector_type}板块列表为空，跳过分析")
            return []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.analyze_single_sector, sector_name, sector_type, days): sector_name
                for sector_name in sectors["板块名称"]
            }

            results = []
            for future in as_completed(futures):
                if (result := future.result()) is not None:
                    results.append(result)

        return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

    # ---------热门龙头----------
    @staticmethod
    def calculate_recent_gain(hist: pd.DataFrame, days) -> float:
        """计算近期涨幅（安全版）"""
        hist = hist.sort_values('日期', ascending=True).tail(days + 1)
        required_days = days + 1  # N日涨幅需要N+1个数据点

        if len(hist) < required_days:
            logger.warning(f"数据不足{required_days}天，当前{len(hist)}天")
            return 0.0

        try:
            # 使用正向索引避免负索引越界
            start_idx = len(hist) - required_days
            end_idx = len(hist) - 1

            start_price = hist['收盘'].iloc[start_idx]
            end_price = hist['收盘'].iloc[end_idx]

            return round((end_price / start_price - 1) * 100, 2)
        except KeyError:
            logger.error("收盘价字段不存在")
            return 0.0
        except IndexError as e:
            logger.error(f"索引计算错误: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_resist_days(stock_hist: pd.DataFrame, index_hist: pd.DataFrame, days) -> int:
        """计算逆势天数"""
        stock_hist = stock_hist.sort_values('日期', ascending=True).tail(days)

        if stock_hist.empty or index_hist.empty:
            return 0

        stock_hist['date'] = pd.to_datetime(stock_hist['日期'])
        index_hist['date'] = pd.to_datetime(index_hist['date'])

        merged = pd.merge(stock_hist, index_hist, on='date', how='inner')
        # 计算逆势天数（个股涨，指数跌）
        resist_days = ((merged['涨跌幅'] > 0) & (merged['pct_change'] < 0)).sum()
        return resist_days

    def calculate_limit_up_info(self, code: str) -> Dict:
        """计算个股涨停信息"""
        zt_df = self.fetch_zt_pool()
        if zt_df.empty:
            return {
                'zt_flag': False,
                'break_times': 0,
                'limit_up_stats': '0天0板',
                'consecutive_times': 0
            }

        # 确保 '代码' 列是字符串类型
        zt_df['代码'] = zt_df['代码'].astype(str).str.zfill(6)

        # 筛选特定股票的数据
        stock_zt_df = zt_df[zt_df['代码'] == code]

        if stock_zt_df.empty:
            return {
                'zt_flag': False,
                'break_times': 0,
                'limit_up_stats': '0天0板',
                'consecutive_times': 0
            }

        # 获取最新的涨停记录
        latest_record = stock_zt_df.iloc[-1]

        # 提取涨停信息
        is_limit_up = True
        break_board_times = latest_record.get('炸板次数', 0)
        limit_up_stats = latest_record.get('涨停统计', '0天0板')
        consecutive_times = latest_record.get('连板数', 0)

        return {
            'zt_flag': is_limit_up,
            'break_times': break_board_times,
            'limit_up_stats': limit_up_stats,
            'consecutive_times': consecutive_times
        }

    @staticmethod
    def calculate_volume_score(hist: pd.DataFrame, days: int = 5) -> float:
        """
        优化点：
        1. 向量化计算替代循环
        2. 动态阈值调整机制
        3. 稳健统计量处理异常值
        4. 时间衰减因子
        5. 边界条件处理
        """
        # 数据校验
        if len(hist) < days or '成交量' not in hist.columns:
            return 0.0

        # 截取最近days日数据（确保数据时效性）
        recent_hist = hist.iloc[-days:].copy()

        try:
            # 使用四分位距替代标准差（抗异常值）
            q3 = recent_hist['成交量'].quantile(0.75)
            q1 = recent_hist['成交量'].quantile(0.25)
            iqr = q3 - q1
            median_vol = recent_hist['成交量'].median()

            # 动态阈值计算（基于市场波动水平）
            vol_increase_threshold = 1.25 + (0.05 * (iqr / median_vol if median_vol > 0 else 1))
            volatility_threshold = 1.0 + (0.5 * (iqr / median_vol if median_vol > 0 else 1))

            # 向量化计算放量标志
            recent_hist['prev_vol'] = recent_hist['成交量'].shift(1)
            condition = (
                    (recent_hist['成交量'] > vol_increase_threshold * recent_hist['prev_vol']) &
                    (recent_hist['成交量'] > median_vol + volatility_threshold * iqr))

            # 生成放量序列（排除首日无前一天数据的情况）
            flags = condition.fillna(False).astype(int).tolist()

            # 计算加权连续天数（指数衰减）
            max_streak = current_streak = 0
            decay_factor = 0.9  # 近期连续更有价值

            for i in range(1, len(flags)):
                if flags[i] == 1:
                    # 连续天数加权（越近期权重越高）
                    current_streak = current_streak * decay_factor + 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0

            # 非线性标准化（S型曲线处理）
            volume = np.exp(-0.5 * (max_streak - days / 2))

            return round(volume, 2)

        except Exception as e:
            logger.error(f"成交量得分计算异常: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_volume_trend(hist: pd.DataFrame, days: int = 5) -> float:
        """计算量能趋势原始值（不涉及评分逻辑）"""
        try:
            if len(hist) < days:
                return 0.0

            # 计算量能趋势强度
            vol_data = hist['成交量'].iloc[-days:]
            slope = (vol_data.values[-1] - vol_data.values[0]) / days
            volatility = vol_data.std()

            # 标准化趋势强度
            return slope / (volatility + 1e-6)  # 防止除零
        except Exception as e:
            logger.error(f"量能趋势计算失败: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_breakthrough(hist: pd.DataFrame, window=20) -> int:
        """计算阶段新高突破次数（修复版）"""
        try:
            if len(hist) < window:
                return 0

            # 1. 计算滚动窗口最高价
            hist['highest'] = hist['最高'].rolling(window, min_periods=window).max()

            # 2. 生成突破标记（收盘价 > 前一日highest）
            breakthroughs = (
                    (hist['收盘'] > hist['highest'].shift(1))  # 逐个比较
                    & (hist['highest'].shift(1).notna()  # 排除初始NaN值
                       )
            )

            # 3. 统计突破次数
            return breakthroughs.astype(int).sum()

        except Exception as e:
            logger.error(f"突破次数计算失败: {str(e)}")
            return 0

    @staticmethod
    def calculate_volume_ratio(hist: pd.DataFrame, days=5) -> float:
        """计算量比指标"""
        hist = hist.sort_values('日期', ascending=False).head(days)

        try:
            if len(hist) < days:
                return 0.0
            # 计算最近days日平均成交量（排除当日）
            mean_vol = hist['成交量'].iloc[-days - 1:-1].mean()
            current_vol = hist['成交量'].iloc[-1]
            return current_vol / mean_vol if mean_vol > 0 else 0.0
        except Exception as e:
            logger.error(f"量比计算失败: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_stock_score(stock_data: Dict) -> float:
        """综合评分逻辑（百分制）"""
        # 突破形态得分（技术面）
        breakthrough_score = SectorConfig.WEIGHTS_CONFIG['breakthrough']['calc'](
            stock_data.get('breakthrough_count', 0))
        # 资金流入得分（资金面）
        inflow_score = SectorConfig.WEIGHTS_CONFIG['funds_inflow']['calc'](
            stock_data.get('fund_inflow_ratio', 0))
        # 逆势抗跌得分（防御性）
        resist_score = SectorConfig.WEIGHTS_CONFIG['resist_days']['calc'](
            stock_data.get('resist_days', 0))
        # 流通市值得分（流动性溢价）
        mv_score = SectorConfig.WEIGHTS_CONFIG['float_mv']['calc'](
            stock_data.get('float_mv', 2e10))
        # 连板数得分（人气指标）
        consecutive_score = SectorConfig.WEIGHTS_CONFIG['consecutive']['calc'](
            stock_data.get('consecutive_times', 0))
        # 近期涨幅得分（价格动量）
        recent_gain_score = SectorConfig.WEIGHTS_CONFIG['recent_gain']['calc'](
            stock_data.get('recent_gain', 0)
        )
        # 成交量得分（市场活跃度）量价配合得分（趋势确认）
        volume_ratio_score = SectorConfig.WEIGHTS_CONFIG['volume_ratio_score']['calc'](
            stock_data.get('volume_ratio', 0)
        )

        # 计算各部分得分
        total_score = (
                breakthrough_score +
                inflow_score +
                resist_score +
                mv_score +
                consecutive_score +
                volume_ratio_score +
                recent_gain_score
        )

        # 归一化到0-100分
        # total_score = total_score / 100

        # 炸板惩罚（每次-3分）
        total_score -= stock_data.get('break_times', 0) * 3

        # 确保分数范围在0-100之间
        total_score = max(min(total_score, 100), 0)

        # 将明细存储到stock_data中
        stock_data['breakthrough_score'] = breakthrough_score
        stock_data['inflow_score'] = inflow_score
        stock_data['resist_score'] = resist_score
        stock_data['mv_score'] = mv_score
        stock_data['consecutive_score'] = consecutive_score
        stock_data['volume_ratio_score'] = volume_ratio_score
        stock_data['recent_gain_score'] = recent_gain_score

        return total_score

    # --------------------------
    # 辅助方法
    # --------------------------
    def _fetch_recent_trade_date(self) -> str:
        """获取最近交易日"""
        return self.calendar.get_recent_trade_date()

    # --------------------------
    # 主流程方法
    # --------------------------
    def generate_hot_sectors(self, days: int = None, top_n_per_type: int = None) -> List[Tuple[str, str, float]]:
        """获取热门板块排行主流程"""
        days = days or SectorConfig.DAYS
        top_n = top_n_per_type or SectorConfig.TOP_N_PER_TYPE

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_ind = executor.submit(self.analyze_sector_data, "industry", days, top_n)
            future_con = executor.submit(self.analyze_sector_data, "concept", days, top_n)
            industry = future_ind.result()
            concept = future_con.result()

        combined = industry + concept
        return sorted(combined, key=lambda x: x[2], reverse=True)[:top_n * 2]

    def generate_sector_dragons(self, sector: str, sector_type: str) -> List[Dict]:
        """稳健版龙头股分析"""
        components = self.fetch_sector_components(sector, sector_type)
        if not components:
            logger.warning(f"板块 {sector} 没有成分股")
            return []

        index_data = self.fetch_index_data()
        if index_data.empty:
            logger.warning("市场指数数据为空")
            return []

        results = []
        start_date = self.calendar.get_previous_trade_date(days=SectorConfig.HIST_DAYS - 1)
        end_date = self.calendar.get_recent_trade_date()

        for code in components:
            # 获取个股历史数据
            stock_hist = self.fetch_sector_stock_hist(symbol=code, start_date=start_date, end_date=end_date)
            if stock_hist.empty:
                continue

            # 获取流通市值（单位：元）
            float_mv = self.fetch_float_market_value(code)

            # 获取资金流入率
            fund_inflow_ratio = self.fetch_fund_inflow(code)

            # 计算突破次数（20日窗口）
            breakthrough_count = self.calculate_breakthrough(stock_hist, SectorConfig.DAYS20)

            # 计算量比（5日平均）
            volume_ratio = self.calculate_volume_ratio(stock_hist, SectorConfig.RESIST_DAYS)

            # 计算各项指标
            recent_gain = self.calculate_recent_gain(stock_hist, SectorConfig.DAYS)
            resist_days = self.calculate_resist_days(stock_hist, index_data, SectorConfig.RESIST_DAYS)
            limit_up_info = self.calculate_limit_up_info(code)

            # 计算综合得分
            stock_data = {
                'code': code,
                'float_mv': float_mv,  # 新增
                'breakthrough_count': breakthrough_count,  # 新增
                'fund_inflow_ratio': fund_inflow_ratio,  # 新增
                'recent_gain': recent_gain,
                'resist_days': resist_days,
                'zt_flag': limit_up_info['zt_flag'],
                'break_times': limit_up_info['break_times'],
                'limit_up_stats': limit_up_info['limit_up_stats'],
                'consecutive_times': limit_up_info['consecutive_times'],
                'volume_ratio': volume_ratio
            }
            stock_data['total_score'] = self.calculate_stock_score(stock_data)
            results.append(stock_data)

        # 按综合得分排序并返回前5名
        return sorted(results, key=lambda x: x['total_score'], reverse=True)[:5]


if __name__ == "__main__":
    analyzer = SectorAnalyzerDB()

    # 获取热门板块
    hot_sectors = analyzer.generate_hot_sectors()
    print("热门板块:", hot_sectors)

    # 分析龙头股
    for sector_info in hot_sectors:
        dragons = analyzer.generate_sector_dragons(sector_info[0], sector_info[1])
        print(f"\n{sector_info[0]} 龙头股:")
        for stock in dragons:
            print(f"""
            股票代码：{stock['code']}
            综合得分：{stock['total_score']:.2f}
            突破形态得分：{stock.get('breakthrough_score', 0):.2f}
            资金流入得分：{stock.get('inflow_score', 0):.2f}
            逆势抗跌得分：{stock.get('resist_score', 0):.2f}
            流通市值得分：{stock.get('mv_score', 0):.2f}
            量价配合得分：{stock.get('volume_ratio_score', 0):.2f}
            连板数得分：{stock.get('consecutive_score', 0):.2f}
            炸板惩罚分数：{-stock.get('break_times', 0) * 3:.2f}
            {SectorConfig.DAYS}日涨幅：{stock['recent_gain']:.2f}%
            涨幅得分：{stock['recent_gain_score']:.2f}
            {SectorConfig.RESIST_DAYS}日逆势天数：{stock['resist_days']}
            炸板次数：{stock['break_times']}
            涨停统计：{stock['limit_up_stats']}
            连板数：{stock['consecutive_times']}
            """)
