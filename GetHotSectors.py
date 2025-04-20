import akshare as ak
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from GetTradeDate import LocalTradeCalendar
from FetchBaseData import DataPathManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
pd.set_option("display.max_columns", None)


# ========================
# 配置类
# ========================
class SectorConfig:
    """板块分析配置参数"""

    # 数据文件名配置
    DATA_FILES = {
        'zt_pool': 'zt_pool.csv',
        'sector_list': 'sector_list.csv',
        'sector_hist': 'sector_hist.csv',
        'lhb_main': 'lhb_main.csv',
        'lhb_detail': 'lhb_detail.csv',
        'index_data': 'index_data.csv',
        'market_breadth': 'market_breadth.csv',
        'market_volume': 'market_volume.csv',
        'zt_stock_hist': 'zt_stock_hist.csv',
        'zt_stock_info': 'zt_stock_info.csv',
        'sector_components': 'sector_components.csv'
    }

    # 分析参数
    DAYS = 3  # 分析周期
    RESIST_DAYS = 5
    TOP_N_PER_TYPE = 5  # 每类板块取前N名
    INDEX_CODE = 'sh000001'  # 基准指数代码

    # 权重配置
    WEIGHTS_CONFIG = {
        'breakthrough': {  # 突破形态得分
            'weight': 25,
            'calc': lambda x: min(x * 5, 25)  # 每突破一次关键位得5分
        },
        'funds_inflow': {  # 资金流入得分
            'weight': 20,
            'calc': lambda x: min(abs(x) * 20, 20)  # 净流入率映射
        },
        'resist_days': {  # 逆势上涨
            'weight': 20,
            'calc': lambda x: min(x * 4, 20)  # 每天4分
        },
        'float_mv': {  # 流通市值
            'weight': 15,
            'calc': lambda x: 15 * (1 - np.log(x + 1) / np.log(2e10))  # 非线性映射
        },
        'volume_ratio': {  # 量价配合
            'weight': 10,
            'calc': lambda x: min(x * 2, 10)  # 量比>2得满分
        },
        'consecutive': {  # 连板数
            'weight': 10,
            'calc': lambda x: min(x * 2, 10)
        }
    }
    VOLUME_SCALE = 3e7  # 成交量标准化基数


# ========================
# 核心分析类
# ========================
class SectorAnalyzer:
    """板块分析核心类"""

    def __init__(self, data_root: str, trade_date: str = None):
        """
        初始化分析器
        :param data_root: 数据存储根目录
        :param trade_date: 交易日（格式：YYYYMMDD），默认取最近交易日
        """
        self.data_mgr = DataPathManager(data_root)
        self.calendar = LocalTradeCalendar()
        self.trade_date = trade_date or self._fetch_recent_trade_date()

    # --------------------------
    # 数据获取方法
    # --------------------------
    def fetch_sector_list(self, sector_type: str) -> pd.DataFrame:
        """获取板块列表（行业/概念）并进行缓存
        Args:
            sector_type: 板块类型 ('industry'/'concept')
        Returns:
            pd.DataFrame: 板块列表数据，包含['板块名称', '成分股数量']等列
        """
        file_name = SectorConfig.DATA_FILES['sector_list']
        df = self.data_mgr.load_data(file_name)
        if not df.empty:
            # logger.info(f"从本地文件加载板块列表: {file_name}")
            df = df[~df["板块名称"].str.contains(r'昨日', case=False, regex=True)]
            return df

        # 如果本地文件不存在，通过 Akshare 获取数据
        try:
            if sector_type == "industry":
                df = ak.stock_board_industry_name_em()
            elif sector_type == "concept":
                df = ak.stock_board_concept_name_em()
                # 清洗概念板块名称中的空格
                df["板块名称"] = df["板块名称"].str.replace(r"\s+", "", regex=True)
            else:
                logger.error(f"不支持的板块类型: {sector_type}")
                return pd.DataFrame()

            df = df[~df["板块名称"].str.contains(r'昨日', case=False, regex=True)]
            return df
        except Exception as e:
            logger.error(f"获取{sector_type}板块列表失败: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def fetch_sector_history(self, sector: str, sector_type: str, days: int) -> pd.DataFrame:
        """获取板块历史数据"""
        file_name = SectorConfig.DATA_FILES['sector_hist']
        df = self.data_mgr.load_data(file_name)
        if not df.empty:
            # logger.info(f"从本地文件加载板块历史数据: {file_name}")
            # 筛选特定板块和类型的数据
            df = df[(df['板块名称'] == sector) & (df['板块类型'] == sector_type)]
            return df

        # 如果本地文件不存在，通过 Akshare 获取数据
        try:
            if sector_type == "industry":
                df = ak.stock_board_industry_hist_em(
                    symbol=sector,
                    period="日k",
                    start_date=self._calculate_start_date(days),
                    end_date=self.trade_date
                )
            elif sector_type == "concept":
                df = ak.stock_board_concept_hist_em(
                    symbol=sector,
                    period="daily",
                    start_date=self._calculate_start_date(days),
                    end_date=self.trade_date
                )
            else:
                return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"获取{sector_type}板块历史数据失败: {str(e)}")
            return pd.DataFrame()

    def fetch_zt_pool(self) -> pd.DataFrame:
        """获取涨停池数据"""
        file_name = SectorConfig.DATA_FILES['zt_pool']
        df = self.data_mgr.load_data(file_name)
        if not df.empty:
            # logger.info(f"从本地文件加载涨停池数据: {file_name}")
            return df

        # 如果本地文件不存在，通过 Akshare 获取数据
        try:
            df = ak.stock_zt_pool_em(date=self.trade_date)
            return df
        except Exception as e:
            logger.error(f"获取涨停池数据失败: {str(e)}")
            return pd.DataFrame()

    def fetch_sector_components(self, sector: str, sector_type: str) -> List[str]:
        """获取板块成分股列表"""
        file_name = SectorConfig.DATA_FILES['sector_components']
        df = self.data_mgr.load_data(file_name)
        if not df.empty:
            # 确保 '代码' 列是字符串类型
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            # 过滤 B 股
            df = df[~df['代码'].str.startswith(('900', '20'))]
            # 筛选特定板块和类型的数据
            df = df[(df['板块名称'] == sector) & (df['板块类型'] == sector_type)]
            return sorted(df['代码'].tolist())

        try:
            if sector_type == "industry":
                df = ak.stock_board_industry_cons_em(symbol=sector)
            elif sector_type == "concept":
                df = ak.stock_board_concept_cons_em(symbol=sector)
            else:
                logger.error(f"不支持的板块类型: {sector_type}")
                return []

            # 确保 '代码' 列是字符串类型
            df['代码'] = df['代码'].astype(str).str.zfill(6)
            # 过滤 B 股
            df = df[~df['代码'].str.startswith(('900', '20'))]
            return sorted(df['代码'].tolist())
        except Exception as e:
            logger.error(f"获取板块成分股失败: {str(e)}")
            return []

    def fetch_sector_stock_hist(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取个股历史数据"""
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start_date, end_date=end_date,
                adjust="qfq"
            )
        except Exception as e:
            logger.error(f"获取股票{symbol}历史数据失败: {str(e)}")
            return pd.DataFrame()

    def fetch_index_data(self) -> pd.DataFrame:
        """获取市场指数数据"""
        file_name = SectorConfig.DATA_FILES['index_data']
        df = self.data_mgr.load_data(file_name)
        if not df.empty:
            df = df[df["指数代码"] == 'sh000001'].tail(10)
            return df
        try:
            df = ak.stock_zh_index_daily(symbol=SectorConfig.INDEX_CODE)
            df['date'] = pd.to_datetime(df['date'])
            df['pct_change'] = df['close'].pct_change() * 100
            return df[['date', 'pct_change']]
        except Exception as e:
            logger.error(f"获取市场指数数据失败: {str(e)}")
            return pd.DataFrame()

    def fetch_float_market_value(self, code: str) -> float:
        """获取流通市值(单位：元)"""
        try:
            file_name = SectorConfig.DATA_FILES['market_volume']
            df = self.data_mgr.load_data(file_name)
            if df.empty:
                logger.warning(f"流通市值数据为空: {code}")
                return 2e10  # 默认返回20亿

            df['代码'] = df['代码'].astype(str).str.zfill(6)
            df['流通市值'] = df['流通市值'].fillna(0)
            filtered_df = df[df['代码'] == code]

            if filtered_df.empty:
                logger.warning(f"未找到代码 {code} 的流通市值数据")
                return 2e10  # 默认返回20亿

            float_mv = filtered_df['流通市值'].iloc[0] / 1e8  # 确保返回一个数值

            return float(float_mv)

        except Exception as e:
            logger.error(f"获取流通市值失败 {code}: {str(e)}")
            return 2e10  # 默认返回20亿

    def fetch_fund_inflow(self, code: str) -> float:
        """获取主力资金净流入率"""
        try:
            file_name = SectorConfig.DATA_FILES['market_volume']
            df = self.data_mgr.load_data(file_name)
            if df.empty:
                logger.warning(f"市场数据为空: {code}")
                return 0.0

            df['代码'] = df['代码'].astype(str).str.zfill(6)
            # 确保 market 是一个单一值
            market_series = df[df['代码'] == code]['市场类型']
            if market_series.empty:
                logger.warning(f"未找到代码 {code} 的市场类型数据")
                return 0.0

            market = market_series.iloc[0]  # 获取市场类型的第一个值

            # 获取最近交易日数据
            fund_flow_df = ak.stock_individual_fund_flow(
                stock=code,
                market=market
            )

            if fund_flow_df.empty:
                logger.warning(f"未找到代码 {code} 的资金流入数据")
                return 0.0

            # 优先使用主力净流入占比
            main_inflow_pct = fund_flow_df['主力净流入-净占比'].iloc[-1]  # 单位：%

            # 辅助验证超大单净流入
            super_inflow = fund_flow_df['超大单净流入-净额'].iloc[-1]

            # 数据合理性校验
            if abs(main_inflow_pct) > 30:  # 异常值过滤
                close_price = fund_flow_df['收盘价'].iloc[-1]
                if close_price > 0:
                    return (super_inflow / close_price) / 1e4  # 使用超大单净额折算
                else:
                    return 0.0

            return main_inflow_pct

        except Exception as e:
            logger.error(f"资金流入数据获取失败 {code}: {str(e)}")
            return 0.0

    # --------------------------
    # 分析计算方法
    # --------------------------
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

    @staticmethod
    def calculate_trend_score(hist_data: pd.DataFrame, n_days: int) -> float:
        """
        计算板块趋势得分（基于N日行情）
        评分规则：
        - 价格动量（65%）：N日累计涨跌幅
        - 量能趋势（35%）：(当前成交量 - N日均量)/N日均量
        """
        try:
            recent_data = hist_data.iloc[-n_days - 1:] if len(hist_data) > n_days else hist_data
            price_change = recent_data['涨跌幅'].sum()
            mean_volume = recent_data['成交量'].mean()
            current_volume = hist_data.iloc[-1]['成交量'] if len(hist_data) > 0 else 0
            volume_ratio = (current_volume - mean_volume) / mean_volume if mean_volume > 0 else 0

            return round(price_change * 0.65 + volume_ratio * 100 * 0.35, 2)
        except Exception as e:
            logger.error(f"趋势评分计算失败: {str(e)}")
            return 0.0

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

    @staticmethod
    def calculate_recent_gain(hist: pd.DataFrame) -> float:
        """计算近期涨幅"""
        if len(hist) < 2:
            return 0.0
        return (hist['收盘'].iloc[-1] / hist['收盘'].iloc[-SectorConfig.DAYS-1] - 1) * 100

    @staticmethod
    def calculate_resist_days(stock_hist: pd.DataFrame, index_hist: pd.DataFrame) -> int:
        """计算逆势天数"""
        if stock_hist.empty or index_hist.empty:
            return 0

        stock_hist['date'] = pd.to_datetime(stock_hist['日期'])
        index_hist['date'] = pd.to_datetime(index_hist['date'])

        # 计算市场指数的涨跌幅
        index_hist['index_pct_change'] = index_hist['close'].pct_change() * 100

        merged = pd.merge(stock_hist, index_hist, on='date', how='inner')
        # 计算逆势天数（个股涨，指数跌）
        resist_days = ((merged['涨跌幅'] > 0) & (merged['index_pct_change'] < 0)).sum()
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
        优化版成交量得分计算（基于动态阈值和加权连续放量）

        优化点：
        1. 向量化计算替代循环
        2. 动态阈值调整机制
        3. 稳健统计量处理异常值
        4. 时间衰减因子
        5. 边界条件处理

        :param hist: 历史数据需包含['成交量']列，并按日期升序排列
        :param days: 统计周期（建议5-10天）
        :return: 标准化得分（0到1之间）
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
            score = 1 / (1 + np.exp(-0.5 * (max_streak - days / 2))) * 100

            return round(score, 2)

        except Exception as e:
            logger.error(f"成交量得分计算异常: {str(e)}")
            return 0.0

    def calculate_breakthrough(self, hist: pd.DataFrame, window=20) -> int:
        """计算阶段新高突破次数"""
        try:
            if len(hist) < window:
                return 0
            # 计算滚动窗口最高价
            hist['highest'] = hist['最高'].rolling(window, min_periods=1).max()
            # 标记突破日（收盘价创新高）
            breakthroughs = hist['收盘'] >= hist['highest'].shift(1).sum()
            return int(breakthroughs)
        except Exception as e:
            logger.error(f"突破次数计算失败: {str(e)}")
            return 0

    def calculate_stock_score(self, stock_data: Dict) -> float:
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
        # 量价配合得分（趋势确认）
        vol_score = SectorConfig.WEIGHTS_CONFIG['volume_ratio']['calc'](
            stock_data.get('volume_ratio', 0))
        # 连板数得分（人气指标）
        consecutive_score = SectorConfig.WEIGHTS_CONFIG['consecutive']['calc'](
            stock_data.get('consecutive_times', 0))
        # 成交量得分（市场活跃度）
        volume_score = stock_data.get('volume_score', 0)
        # 近期涨幅得分（价格动量）
        recent_gain = stock_data.get('recent_gain', 0)

        # 权重配置（可以根据需要调整）
        weights = {
            'breakthrough': 15,  # 突破形态权重
            'funds_inflow': 15,  # 资金流入权重
            'resist_days': 10,  # 逆势抗跌权重
            'float_mv': 10,  # 流通市值权重
            'volume_ratio': 10,  # 量价配合权重
            'consecutive': 10,  # 连板数权重
            'volume_score': 15,  # 成交量得分权重
            'recent_gain': 25  # 近期涨幅权重
        }

        # 计算各部分得分
        total_score = (
                breakthrough_score * weights['breakthrough'] +
                inflow_score * weights['funds_inflow'] +
                resist_score * weights['resist_days'] +
                mv_score * weights['float_mv'] +
                vol_score * weights['volume_ratio'] +
                consecutive_score * weights['consecutive'] +
                volume_score * weights['volume_score'] +
                recent_gain * weights['recent_gain']
        )

        # 归一化到0-100分
        total_score = total_score / 100

        # 炸板惩罚（每次-3分）
        total_score -= stock_data.get('break_times', 0) * 3

        # 确保分数范围在0-100之间
        total_score = max(min(total_score, 100), 0)

        # 将明细存储到stock_data中
        stock_data['breakthrough_score'] = breakthrough_score
        stock_data['inflow_score'] = inflow_score
        stock_data['resist_score'] = resist_score
        stock_data['mv_score'] = mv_score
        stock_data['vol_score'] = vol_score
        stock_data['consecutive_score'] = consecutive_score
        stock_data['volume_score'] = volume_score
        stock_data['recent_gain_score'] = recent_gain

        return total_score

    def calculate_volume_ratio(self, hist: pd.DataFrame, days=5) -> float:
        """计算量比指标"""
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
    # --------------------------
    # 辅助方法
    # --------------------------
    def _fetch_recent_trade_date(self) -> str:
        """获取最近交易日"""
        return self.calendar.get_recent_trade_date()

    def _calculate_start_date(self, days: int) -> str:
        """计算起始日期"""
        return self.calendar.get_previous_trade_date(base_date=self.trade_date, days=days)

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
        start_date = self.calendar.get_previous_trade_date(days=SectorConfig.RESIST_DAYS)
        end_date = self.calendar.get_recent_trade_date()

        for code in components:
            # 获取个股历史数据
            stock_hist = self.fetch_sector_stock_hist(symbol=code, start_date=start_date, end_date=end_date)
            if stock_hist.empty:
                continue

            # 获取流通市值（单位：元）
            float_mv = self.fetch_float_market_value(code)
            # 计算突破次数（20日窗口）
            breakthrough_count = self.calculate_breakthrough(stock_hist)
            # 获取资金流入率
            fund_inflow_ratio = self.fetch_fund_inflow(code)
            # 计算量比（5日平均）
            volume_ratio = self.calculate_volume_ratio(stock_hist)

            # 计算各项指标
            recent_gain = self.calculate_recent_gain(stock_hist)
            resist_days = self.calculate_resist_days(stock_hist, index_data)
            limit_up_info = self.calculate_limit_up_info(code)
            volume_score = self.calculate_volume_score(stock_hist)

            # 计算综合得分
            stock_data = {
                'code': code,
                'float_mv': float_mv,  # 新增
                'breakthrough_count': breakthrough_count,  # 新增
                'fund_inflow_ratio': fund_inflow_ratio,  # 新增
                'volume_ratio': volume_ratio,  # 新增
                'recent_gain': recent_gain,
                'resist_days': resist_days,
                'zt_flag': limit_up_info['zt_flag'],
                'break_times': limit_up_info['break_times'],
                'limit_up_stats': limit_up_info['limit_up_stats'],
                'consecutive_times': limit_up_info['consecutive_times'],
                'volume_score': volume_score
            }
            stock_data['total_score'] = self.calculate_stock_score(stock_data)
            results.append(stock_data)

        # 按综合得分排序并返回前5名
        return sorted(results, key=lambda x: x['total_score'], reverse=True)[:5]


if __name__ == "__main__":
    data_root = "data/csv/20250418"  # 替换为实际的数据存储路径
    analyzer = SectorAnalyzer(data_root)

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
            量价配合得分：{stock.get('vol_score', 0):.2f}
            连板数得分：{stock.get('consecutive_score', 0):.2f}
            炸板惩罚分数：{-stock.get('break_times', 0) * 3:.2f}
            {SectorConfig.DAYS}日涨幅：{stock['recent_gain']:.2f}%
            {SectorConfig.RESIST_DAYS}日逆势天数：{stock['resist_days']}
            是否涨停：{stock['zt_flag']}
            炸板次数：{stock['break_times']}
            涨停统计：{stock['limit_up_stats']}
            连板数：{stock['consecutive_times']}
            成交量得分：{stock['volume_score']:.2f}
            """)
