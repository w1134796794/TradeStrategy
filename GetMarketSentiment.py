import pandas as pd
import logging
import akshare as ak
from GetTradeDate import LocalTradeCalendar
from FetchBaseData import DataPathManager
from typing import Dict
pd.set_option("display.max_columns", None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 配置类 ====================
class MarketSentimentConfig:
    """市场情绪分析配置"""
    index_weights = {"sh000001": 0.4, "sz399001": 0.3, "sz399006": 0.3}
    score_weights = {'index': 0.4, 'breadth': 0.3, 'limit': 0.3, 'premium': 0.5}
    max_retries: int = 3
    retry_delay: float = 1.5
    new_stock_threshold: int = 5  # 新股判定交易日数

    # 数据文件名配置
    DATA_FILES = {
        'index_data': 'index_data.csv',
        'market_breadth': 'market_breadth.csv',
        'market_volume': 'market_volume.csv',
        'zt_pool': 'zt_pool.csv',
        'zt_stock_info': 'zt_stock_info.csv'
    }


# ==================== 核心分析类 ====================
class MarketSentimentAnalyzer:
    """市场情绪分析器（本地文件加载版）"""

    def __init__(self, config: MarketSentimentConfig = None, data_root: str = None):
        self.config = config or MarketSentimentConfig()
        self.data_mgr = DataPathManager(data_root) if data_root else None
        self.calendar = LocalTradeCalendar()
        self.trade_date = self.calendar.get_recent_trade_date()

        # 初始化缓存
        self._index_data: Dict = {}
        self._market_breadth: Dict = {}
        self._limit_stats: Dict = {}
        self._listing_dates = {}

        # 指数名称映射
        self._index_name_map = {
            "sh000001": "上证指数", "sz399001": "深证成指",
            "sz399006": "创业板指", "sh000016": "上证50"
        }

    # ==================== 数据获取方法 ====================
    def fetch_index_data(self) -> Dict:
        """从本地文件加载指数数据"""
        index_data = {}
        for code, weight in self.config.index_weights.items():
            try:
                # 从本地文件加载
                file_name = MarketSentimentConfig.DATA_FILES['index_data']
                df = self.data_mgr.load_data(file_name)

                # 筛选对应的指数数据
                df = df[df["指数代码"] == code]

                if df.empty:
                    logger.warning(f"指数{code}数据为空")
                    continue

                df = df.iloc[-10:]
                df['ma2'] = df['close'].rolling(2).mean()
                df['ma6'] = df['close'].rolling(6).mean()
                last = df.iloc[-1]

                processed = {
                    'change_pct': (last['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100,
                    'position': 'above' if last['close'] > last['ma2'] else 'below',
                    'trend': 'up' if last['ma2'] > last['ma6'] else 'down',
                    'weight': weight
                }

                index_data[code] = processed

            except Exception as e:
                logger.error(f"加载指数{code}数据失败: {str(e)}")
        return index_data

    def fetch_market_breadth(self) -> Dict:
        """获取市场广度数据（带缓存）"""

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
            logger.error(f"市场广度获取失败: {str(e)}")
            return {k: 0 for k in ['rise_num', 'fall_num', 'limit_up', 'limit_down']}

    def fetch_limit_stats(self) -> Dict:
        """从本地文件加载涨停统计数据"""
        try:
            file_name = MarketSentimentConfig.DATA_FILES['zt_pool']
            df = self.data_mgr.load_data(file_name)

            if df.empty:
                logger.warning("涨停统计数据为空")
                return {'max_limit': 0, 'distribution': {}, 'non_consecutive': []}

            # 调试打印前3行数据
            logger.debug("涨停池原始数据样例:\n%s", df.head(3).to_string())

            non_consecutive = []
            for _, row in df.iterrows():
                try:
                    # 解析涨停统计字段
                    if '/' not in str(row['涨停统计']):
                        logger.warning(f"股票{row['代码']}异常涨停统计格式: {row['涨停统计']}")
                        continue

                    total_days, limit_times = map(int, str(row['涨停统计']).split('/'))
                    consecutive_days = row['连板数']

                    # 计算非连续涨停次数
                    if limit_times > consecutive_days:
                        non_consecutive.append({
                            'code': row['代码'],
                            'name': row['名称'],
                            'total_days': total_days,  # 总交易天数
                            'limit_times': limit_times,  # 总涨停次数
                            'consecutive_days': consecutive_days,  # 最大连板
                            'non_consecutive': limit_times - consecutive_days  # 非连续次数
                        })
                except Exception as e:
                    logger.warning(f"处理{row['代码']}时异常: {str(e)}")

            return {
                'max_limit': df['连板数'].max() if not df.empty else 0,
                'distribution': df['连板数'].value_counts().to_dict(),
                'non_consecutive': non_consecutive
            }

        except Exception as e:
            logger.error(f"加载涨停统计数据失败: {str(e)}")
            return {'max_limit': 0, 'distribution': {}, 'non_consecutive': []}

    def fetch_market_volume_data(self, end_date: str, days: int = 5) -> pd.DataFrame:
        """从本地data/csv/yyyymmdd/market_volume.csv中获取指定日期范围的市场成交量数据"""
        date_range = self.calendar.get_recent_trade_dates(end_date, days)
        all_data = []

        for date in date_range:

            file_path = f"data/csv/{date}/market_volume.csv"

            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    # 确保日期格式统一
                    df['成交额'] = df['成交额'].fillna(0)
                    df['日期'] = date
                    # df["日期"] = df["日期"].astype(str).str.replace("-", "", regex=True)
                    all_data.append(df)
            except FileNotFoundError:
                logger.warning(f"文件{file_path}不存在")
            except Exception as e:
                logger.error(f"加载{file_path}失败: {str(e)}")

        if not all_data:
            return pd.DataFrame()

        # 合并所有数据并按日期排序
        combined_df = pd.concat(all_data).sort_values("日期")
        return combined_df

    # ==================== 分析计算方法 ====================
    def calculate_volume_trend(self, end_date: str, days: int = 5) -> Dict:
        """计算指定日期范围内的市场平均成交量，并判断结束日期的成交量状态"""
        # 加载指定日期范围的数据
        data_df = self.fetch_market_volume_data(end_date, days)

        if data_df.empty:
            logger.warning("没有找到指定日期范围内的数据")
            return {
                "average_volume": 0.0,
                "current_volume": 0.0,
                "volume_status": "无数据",
                "change_ratio": 0.0
            }

        # 计算每日总成交量
        daily_volume = data_df.groupby("日期")["成交额"].sum().reset_index()
        daily_volume["成交额"] = daily_volume["成交额"].astype(float) / 1e8

        # 获取最后一天的数据
        end_date_volume = daily_volume.iloc[-1]["成交额"]

        # 计算平均成交额（不包括最后一天）
        average_df = daily_volume.iloc[:-1]
        if average_df.empty:
            logger.warning("计算平均成交额的数据不足")
            return {
                "average_volume": 0.0,
                "current_volume": end_date_volume,
                "volume_status": "数据不足",
                "change_ratio": 0.0
            }

        average_volume = average_df["成交额"].mean()

        # 计算变化率
        change_ratio = (end_date_volume - average_volume) / average_volume if average_volume != 0 else 0

        # 判断成交量状态
        if change_ratio > 0.2:
            volume_status = "明显放量"
        elif change_ratio > 0.1:
            volume_status = "温和放量"
        elif change_ratio < -0.15:
            volume_status = "显著缩量"
        elif change_ratio < -0.05:
            volume_status = "轻微缩量"
        else:
            volume_status = "平量"

        return {
            "average_volume": round(average_volume, 2),
            "current_volume": round(end_date_volume, 2),
            "volume_status": volume_status,
            "change_ratio": round(change_ratio, 4)
        }

    def analyze_premium_effect(self, days: int = 5):
        """分析首板溢价效应（次日开盘买入，第三日开盘卖出）"""
        try:
            profits = []  # 收益率列表
            total_samples = 0  # 总样本数

            date_list = self.calendar.get_recent_trade_dates(self.trade_date, days)
            print(date_list)

            for date in date_list[:-2]:  # 需要确保有次日和第三日
                if not date:
                    logger.warning("遇到无效日期，终止循环")
                    break

                # 加载涨停数据
                zt_df = self._load_zt_pool_data(date)
                if zt_df.empty:
                    logger.warning(f"{date}无涨停数据")
                    continue

                # 确保日期和代码列的数据类型一致
                zt_df["日期"] = zt_df["日期"].astype(str)
                zt_df["代码"] = zt_df["代码"].astype(str).str.zfill(6)

                # 筛选首板
                first_zt = zt_df[zt_df['连板数'] == 2]

                if first_zt.empty:
                    logger.info(f"{date}无首板股票")
                    continue

                # 获取次日和第三日的日期
                next_date = self.calendar.get_next_trade_date(date)
                day_after_next = self.calendar.get_next_trade_date(next_date)
                if not next_date or not day_after_next:
                    logger.info(f"{date}后无有效交易日")
                    continue

                # 遍历首板股票
                for _, row in first_zt.iterrows():
                    code = row['代码']
                    close_price = row['最新价']

                    # 跳过无效收盘价
                    if close_price <= 0:
                        logger.warning(f"股票{code}收盘价异常: {close_price}")
                        continue

                    # 获取次日K线数据
                    next_day_kline = self._load_stock_hist(code, next_date)
                    if next_day_kline.empty:
                        logger.warning(f"股票{code}在{next_date}无数据")
                        continue

                    # 获取第三日K线数据
                    day_after_next_kline = self._load_stock_hist(code, day_after_next)
                    if day_after_next_kline.empty:
                        logger.warning(f"股票{code}在{day_after_next}无数据")
                        continue

                    # 提取次日开盘价和第三日开盘价
                    next_day_open = next_day_kline.iloc[0]['开盘']
                    day_after_next_open = day_after_next_kline.iloc[0]['开盘']

                    if next_day_open <= 0 or day_after_next_open <= 0:
                        logger.warning(f"股票{code}开盘价异常: {next_day_open} 或 {day_after_next_open}")
                        continue

                    # 计算收益率
                    profit_pct = (day_after_next_open / next_day_open - 1) * 100
                    profits.append(profit_pct)
                    total_samples += 1

            # 计算统计指标
            if total_samples == 0:
                return {
                    'profit_probability': 0.0,
                    'avg_profit': 0.0,
                    'avg_loss': 0.0,
                    'winning_profit_rate': 0.0,
                    'total_samples': 0
                }

            # 盈利概率
            profit_probability = sum(1 for p in profits if p > 0) / total_samples * 100

            # 平均收益率
            avg_profit = sum(profits) / total_samples

            # 盈利率
            winning_samples = [p for p in profits if p > 0]
            winning_profit_rate = sum(winning_samples) / len(winning_samples) if winning_samples else 0.0

            # 亏损率（仅统计亏损的样本）
            loss_samples = [p for p in profits if p < 0]
            avg_loss = sum(loss_samples) / len(loss_samples) if loss_samples else 0.0

            logger.info(f"盈利概率: {profit_probability:.2f}%")
            logger.info(f"平均收益率: {avg_profit:.2f}%")
            logger.info(f"亏损率: {avg_loss:.2f}% (基于 {len(loss_samples)} 个亏损样本)")

            return {
                'profit_probability': profit_probability,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'winning_profit_rate': winning_profit_rate,
                'total_samples': total_samples
            }

        except Exception as e:
            logger.error(f"溢价分析失败: {str(e)}", exc_info=True)
            return {
                'profit_probability': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'winning_profit_rate': 0.0,
                'total_samples': 0
            }

    def calculate_total_score(self) -> float:
        """优化版市场情绪评分（集成量能分析）"""
        try:
            total = 0.0
            end_date = self.calendar.get_recent_trade_date()

            # ==================== 量能分析维度 (35%) ====================
            volume_analysis = self.calculate_volume_trend(end_date, 5)

            # 量能基础得分（基于变化率）
            change_ratio = volume_analysis['change_ratio']
            if change_ratio > 0.2:
                vol_base = 25  # 明显放量
            elif change_ratio > 0.1:
                vol_base = 20  # 温和放量
            elif change_ratio < -0.15:
                vol_base = 5  # 显著缩量
            elif change_ratio < -0.05:
                vol_base = 10  # 轻微缩量
            else:
                vol_base = 15  # 平量

            # 量能趋势得分（结合近期趋势）
            trend_score = 0
            if volume_analysis['current_volume'] > volume_analysis['average_volume']:
                # 每高500亿加5分
                trend_score = (volume_analysis['current_volume'] - volume_analysis['average_volume']) / 500 * 5

            total += min((vol_base + trend_score), 35)

            # ==================== 市场热度维度 (50%) ====================
            breadth = self._market_breadth
            valid_stocks = breadth['rise_num'] + breadth['fall_num']

            # 上涨强度得分
            rise_strength = 0
            if valid_stocks > 0:
                rise_ratio = breadth['rise_num'] / valid_stocks
                # 上涨家数比例越高得分越高，最高30分
                rise_strength = min(rise_ratio * 50, 30)

                # 普涨加成（上涨家数>3000）
                if breadth['rise_num'] > 3500:
                    rise_strength += 10

            # 涨停效应得分
            limit_score = breadth['limit_up'] * 0.3  # 每个涨停+0.3
            # 跌停惩罚
            down_penalty = -breadth['limit_down']  # 每个跌停-1

            total += min((rise_strength + limit_score + down_penalty), 50)

            # ==================== 指数协同维度 (最高15分) ====================
            index_gain = sum(data['change_pct'] * data['weight'] for data in self._index_data.values())

            # 指数涨幅基础分
            index_base = min(max(index_gain * 5, 0), 10)  # 1%涨幅=5分，最高10分

            # 量价配合加分
            volume_bonus = 0
            if index_gain > 0.5 and volume_analysis['vol_status'] in ["明显放量", "温和放量"]:
                volume_bonus = 5  # 放量上涨额外加5分

            total += min(index_base + volume_bonus, 15)  # 指数协同维度最高15分

            return min(max(round(total, 1), 0), 100)  # 确保0-100区间

        except Exception as e:
            logger.error(f"评分计算失败: {str(e)}", exc_info=True)
            return 0.0

    # ==================== 主流程方法 ====================
    def generate_report(self) -> Dict:
        """生成完整市场情绪报告（基于本地文件）"""
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

    # ==================== 辅助方法 ====================
    @staticmethod
    def _get_sentiment_label(score: float) -> str:
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

    def _format_index_data(self) -> Dict:
        """格式化指数数据为中文"""
        formatted = {}
        for code, data in self._index_data.items():
            formatted[code] = {
                '名称': self._index_name_map.get(code, "未知指数"),
                '当日涨跌幅': f"{data.get('change_pct', 0.0):.2f}%",
                '均线位置': "5日均线上方" if data.get('position') == 'above' else "5日均线下方",
                '趋势方向': "上升趋势" if data.get('trend') == 'up' else "下降趋势",
                '权重占比': f"{data.get('weight', 0.0) * 100:.1f}%"
            }
        return formatted

    def _load_zt_pool_data(self, date_str: str) -> pd.DataFrame:
        """从本地加载指定日期的涨停数据"""
        file_name = f"zt_pool_hist.csv"
        zt_df = self.data_mgr.load_data(file_name)
        zt_df['代码'] = zt_df['代码'].astype(str).str.zfill(6)
        zt_df["日期"] = zt_df["日期"].astype(str)
        zt_df = zt_df[zt_df["日期"] == date_str]

        return zt_df

    def _load_stock_hist(self, code: str, date_str: str) -> pd.DataFrame:
        """从本地加载指定股票和日期的历史数据"""
        file_name = f"zt_stock_hist.csv"
        df = self.data_mgr.load_data(file_name)
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
        df["日期"] = df["日期"].astype(str).replace('-', '', regex=True)
        df = df[(df["日期"] == date_str) & (df["股票代码"] == code)]

        return df


# ==================== 使用示例 ====================
if __name__ == "__main__":
    data_root = "data/csv/20250418"  # 替换为实际的数据存储路径
    analyzer = MarketSentimentAnalyzer(data_root=data_root)
    report = analyzer.generate_report()

    print(f"\n【市场情绪报告】{report['trade_date']}")
    print(f"综合情绪分: {report['total_score']} ({report['sentiment_level']})")
    print("\n指数数据:")
    for code, data in report['index_data'].items():
        print(f"- {data['名称']}: {data['当日涨跌幅']} | {data['趋势方向']}")

    print("\n涨停分析:")
    print(f"最高连板: {report['limit_stats']['max_limit']} 连板")
    print(f"涨停分布: {report['limit_stats']['distribution']}")

    print(f"\n二板溢价效应分析:")
    print(f"盈利概率: {report['premium_effect']['profit_probability']:.2f}%")
    print(f"平均收益率: {report['premium_effect']['avg_profit']:.2f}%")
    print(f"亏损率: {report['premium_effect']['avg_loss']:.2f}% (基于 {report['premium_effect']['total_samples']} 个样本)")
    print(f"盈利率: {report['premium_effect']['winning_profit_rate']:.2f}% (基于 {report['premium_effect']['total_samples']} 个样本)")

