import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import logging
from dataclasses import dataclass
import time
from TradeDate import TradeCalendar
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketSentimentConfig:
    """市场情绪分析配置参数"""
    # 指数权重配置（总和建议为1）
    index_weights = {
        "sh000001": 0.4,  # 上证指数
        "sz399001": 0.3,  # 深证成指
        "sz399006": 0.3  # 创业板指
    }

    # 评分模型参数
    base_score: float = 50.0  # 基础情绪分
    index_weight: float = 0.4  # 指数维度权重
    breadth_weight: float = 0.3  # 市场广度权重
    limit_weight: float = 0.3  # 连板高度权重

    # 重试参数
    max_retries: int = 3  # 接口调用重试次数
    retry_delay: float = 1.0  # 重试间隔(秒)

    # 量能权重
    volume_weight: float = 0.2


class MarketSentimentAnalyzer:
    """市场情绪分析器（优化版）"""
    def __init__(self, config: MarketSentimentConfig = None):
        self.config = config or MarketSentimentConfig()
        self.trade_date = datetime.now().strftime("%Y%m%d")
        self.calendar = TradeCalendar()

        # 初始化数据缓存
        self._index_data: Optional[Dict] = None
        self._market_breadth: Optional[Dict] = None
        self._limit_stats: Optional[Dict] = None
        self._listing_dates = {}
        self.new_stock_threshold: int = 5  # 上市天数阈值（交易日）

        self.market_amplitude = {
            'main_board': 0.18,  # 主板60/00开头
            'gem': 0.36,  # 创业板30开头
            'star': 0.36,  # 科创板68开头
            'bj': 0.45  # 北交所43/83/87开头
        }

        # 情绪评分参数
        self.extreme_score_config = {
            'sky_earth_penalty': -3,  # 天地板扣分
            'earth_sky_bonus': 2,  # 地天板加分
            'st_penalty_factor': 1.5  # ST股影响系数
        }

        self._index_name_map = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
            "sz399005": "中小板指",
            "sh000016": "上证50",
            "sh000905": "中证500",
            "sh000300": "沪深300"
        }

    def _get_index_name(self, code: str) -> str:
        """获取指数中文名称"""
        return self._index_name_map.get(code, "未知指数")

    def detect_extreme_boards(self) -> Dict:
        """基于实时振幅的极端波动检测"""
        try:
            # 获取全市场实时行情
            spot_df = ak.stock_zh_a_spot_em()
        except Exception as e:
            logger.error(f"实时行情获取失败: {str(e)}")
            return {'sky_earth': 0, 'earth_sky': 0}

        extreme_cases = {'sky_earth': 0, 'earth_sky': 0, 'details': []}

        for _, row in spot_df.iterrows():
            try:
                # 基础数据校验
                if self._is_new_stock(row['代码']):
                    continue

                if pd.isna(row['振幅']) or pd.isna(row['最新价']):
                    continue

                # 获取市场类型
                market = self._get_market_type(row['代码'])
                if market not in self.market_amplitude:
                    continue

                # 获取关键数据
                amplitude = row['振幅'] / 100  # 转换百分比为小数
                last_price = row['最新价']
                prev_close = row['昨收']
                is_st = 'ST' in row['名称']

                # 计算涨跌方向
                price_change = (last_price - prev_close) / prev_close

                # 判断逻辑
                if amplitude >= self.market_amplitude[market]:
                    # 地天板条件：振幅达标且最新价高于昨日收盘
                    if price_change > 0:
                        extreme_cases['earth_sky'] += 1
                        extreme_cases['details'].append({
                            'code': row['代码'],
                            'name': row['名称'],
                            'type': '地天板',
                            'amplitude': amplitude,
                            'change_pct': price_change * 100
                        })
                    # 天地板条件：振幅达标且最新价低于昨日收盘
                    else:
                        extreme_cases['sky_earth'] += 1
                        extreme_cases['details'].append({
                            'code': row['代码'],
                            'name': row['名称'],
                            'type': '天地板',
                            'amplitude': amplitude,
                            'change_pct': price_change * 100
                        })
            except Exception as e:
                logger.warning(f"处理{row['代码']}时异常: {str(e)}")

        return extreme_cases

    def _get_listing_date(self, symbol: str) -> Optional[datetime]:
        """获取上市日期（带缓存）"""
        if symbol not in self._listing_dates:
            try:
                # 获取股票基本信息
                df = ak.stock_individual_info_em(symbol=symbol)

                # 提取上市日期字段
                date_row = df[df['item'] == '上市时间']
                if date_row.empty:
                    logger.warning(f"股票{symbol}无上市日期信息")
                    return None

                # 处理不同数据格式
                raw_date = date_row['value'].iloc[0]

                # 类型转换和格式处理
                if isinstance(raw_date, int):  # 处理数字格式日期
                    date_str = str(raw_date)
                    if len(date_str) == 8:  # 格式如20230830
                        return datetime.strptime(date_str, "%Y%m%d")
                    else:  # 处理其他数字格式
                        logger.warning(f"股票{symbol}异常日期格式: {raw_date}")
                        return None
                elif isinstance(raw_date, str):  # 处理字符串格式
                    # 尝试多种日期格式解析
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
                        try:
                            return datetime.strptime(raw_date, fmt)
                        except ValueError:
                            continue
                    logger.warning(f"股票{symbol}无法解析的日期格式: {raw_date}")
                    return None
                else:  # 未知类型
                    logger.warning(f"股票{symbol}日期字段类型异常: {type(raw_date)}")
                    return None

            except Exception as e:
                logger.error(f"获取{symbol}上市日期失败: {str(e)}")
                return None
        return self._listing_dates[symbol]

    def _is_new_stock(self, symbol: str) -> bool:
        """判断是否为新股次新股"""
        listing_date = self._get_listing_date(symbol)
        if not listing_date:
            return False  # 获取失败时不排除

        # 计算实际交易日差
        trade_days = self.calendar.get_trade_days(
            start_date=listing_date.strftime("%Y%m%d"),
            end_date=self.trade_date
        )
        return len(trade_days) <= self.new_stock_threshold

    def calculate_extreme_score(self, extreme_data: Dict) -> float:
        """计算极端波动得分"""
        score = 0
        # 基础得分计算
        score += extreme_data['sky_earth'] * self.extreme_score_config['sky_earth_penalty']
        score += extreme_data['earth_sky'] * self.extreme_score_config['earth_sky_bonus']

        # ST股额外惩罚
        st_count = sum(1 for d in extreme_data['details'] if 'ST' in d['name'])
        score *= self.extreme_score_config['st_penalty_factor'] ** st_count

        return score

    def _get_market_type(self, code: str) -> str:
        """市场类型识别"""
        prefix_map = {
            '60': 'main_board',
            '00': 'main_board',
            '30': 'gem',
            '68': 'star',
            '43': 'bj',
            '83': 'bj',
            '87': 'bj'
        }
        return prefix_map.get(code[:2], 'unknown')

    def _safe_api_call(self, api_func, *args, **kwargs):
        """带重试机制的API调用"""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return api_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"接口调用第{attempt}次失败: {str(e)}")
                if attempt == self.config.max_retries:
                    raise
                time.sleep(self.config.retry_delay)

    def fetch_index_data(self) -> Dict:
        """获取并处理指数数据"""
        index_data = {}

        for index_code, weight in self.config.index_weights.items():
            try:
                df = self._safe_api_call(ak.stock_zh_index_daily, symbol=index_code)
                if df.empty:
                    continue

                # 计算技术指标
                df = df.iloc[-30:]  # 保留最近30个交易日
                df['ma5'] = df['close'].rolling(5).mean()
                df['ma20'] = df['close'].rolling(20).mean()
                last = df.iloc[-1]

                index_data[index_code] = {
                    'change_pct': (last['close'] - last['open']) / last['open'] * 100,
                    'position': 'above' if last['close'] > last['ma5'] else 'below',
                    'trend': 'up' if last['ma5'] > last['ma20'] else 'down',
                    'weight': weight
                }

            except Exception as e:
                logger.error(f"指数{index_code}数据处理失败: {str(e)}")

        return index_data

    def fetch_market_breadth(self) -> Dict:
        """获取市场广度数据"""
        try:
            df = self._safe_api_call(ak.stock_market_activity_legu)
            breadth_items = {
                'rise_num': ('上涨', int),
                'fall_num': ('下跌', int),
                'limit_up': ('真实涨停', int),
                'limit_down': ('真实跌停', int)
            }

            result = {}
            for key, (name, dtype) in breadth_items.items():
                try:
                    value = df[df['item'] == name]['value'].iloc[0]
                    result[key] = dtype(value)
                except (IndexError, KeyError, ValueError) as e:
                    logger.warning(f"市场广度字段[{name}]获取失败: {str(e)}")
                    result[key] = 0
            return result
        except Exception as e:
            logger.error(f"市场广度数据获取失败: {str(e)}")
            return {k: 0 for k in breadth_items.keys()}

    def fetch_limit_stats(self) -> Dict:
        """精确解析涨停统计数据"""
        try:
            zt_data = ak.stock_zt_pool_em(date=self.trade_date)
            if zt_data.empty:
                return {'max_limit': 0, 'distribution': {}, 'non_consecutive': []}

            # 调试打印前3行数据
            logger.debug("涨停池原始数据样例:\n%s", zt_data.head(3).to_string())

            non_consecutive = []
            for _, row in zt_data.iterrows():
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
                'max_limit': zt_data['连板数'].max(),
                'distribution': zt_data['连板数'].value_counts().to_dict(),
                'non_consecutive': non_consecutive
            }
        except Exception as e:
            logger.error(f"涨停数据获取失败: {str(e)}")
            return {'max_limit': 0, 'distribution': {}, 'non_consecutive': []}

    def _format_limit_stats(self) -> Dict:
        """增强版格式化输出"""
        stats = {
            '最高连板数': self._limit_stats['max_limit'],
            '连板分布': {f"{k}连板": v for k, v in self._limit_stats['distribution'].items()},
            '特殊涨停案例': []
        }

        for case in self._limit_stats['non_consecutive']:
            stats['特殊涨停案例'].append(
                f"{case['name']}({case['code']}): "
                f"{case['total_days']}个交易日内{case['limit_times']}次涨停，"
                f"最长{case['consecutive_days']}连板，"
                f"非连续涨停{case['non_consecutive']}次"
            )

        return stats

    def collect_market_data(self) -> Dict:
        """整合市场数据"""
        self._index_data = self.fetch_index_data()
        self._market_breadth = self.fetch_market_breadth()
        self._limit_stats = self.fetch_limit_stats()

        if self.trade_date:
            prev_date = self.calendar.get_previous_trade_date(self.trade_date)

        return {
            'index_data': self._index_data,
            'market_breadth': self._market_breadth,
            'limit_stats': self._limit_stats,
            'trade_date': self.trade_date
        }

    def calculate_index_score(self) -> float:
        """计算指数维度得分"""
        total_score = 0.0

        for code, data in self._index_data.items():
            weight = data.get('weight', 0)

            # 趋势评分（0-20分）
            trend_score = 20 if data['trend'] == 'up' else 0

            # 位置评分（0-10分）
            position_score = 10 if data['position'] == 'above' else 0

            # 涨跌幅评分（-20~20分）
            change_score = max(min(data['change_pct'] * 2, 20), -20)

            # 加权计算
            total_score += (trend_score + position_score + change_score) * weight

        return total_score * self.config.index_weight

    def calculate_breadth_score(self) -> float:
        """计算市场广度得分"""
        breadth = self._market_breadth

        # 上涨比例得分（0-30分）
        rise_ratio = breadth['rise_num'] / max(breadth['rise_num'] + breadth['fall_num'], 1)
        rise_score = min(rise_ratio * 100, 30)

        # 涨跌停比得分（0-20分）
        limit_ratio = breadth['limit_up'] / max(breadth['limit_down'], 1)
        limit_score = min(np.log1p(limit_ratio) * 10, 20)  # 使用对数压缩量级

        # 涨停数量奖励分
        bonus_score = 10 if breadth['limit_up'] > 50 else 0

        return (rise_score + limit_score + bonus_score) * self.config.breadth_weight

    def calculate_limit_score(self) -> float:
        """计算连板高度得分"""
        stats = self._limit_stats

        # 最高连板得分（0-15分）
        max_score = min(stats['max_limit'] * 5, 15)

        # 连板分布得分（0-10分）
        mid_limit = sum(count for limit, count in stats['distribution'].items() if 3 <= limit < 7)
        distribution_score = 10 if mid_limit > 5 else 5 if mid_limit > 3 else 0

        return (max_score + distribution_score) * self.config.limit_weight

    def calculate_total_score(self) -> float:
        """计算综合情绪得分"""
        if not all([self._index_data, self._market_breadth, self._limit_stats]):
            self.collect_market_data()

        total = self.config.base_score
        total += self.calculate_index_score()
        total += self.calculate_breadth_score()
        total += self.calculate_limit_score()

        return max(min(total, 100), 0)  # 限制在0-100区间

    def generate_report(self) -> Dict:
        """生成分析报告"""
        score = self.calculate_total_score()

        return {
            '交易日期': self.trade_date,
            '综合情绪分': round(score, 1),
            '得分明细': {
                '指数维度得分': round(self.calculate_index_score(), 1),
                '市场广度得分': round(self.calculate_breadth_score(), 1),
                '连板高度得分': round(self.calculate_limit_score(), 1)
            },
            '市场数据': {
                '指数数据': self._format_index_data(),
                '涨跌统计': self._format_market_breadth(),
                '涨停分析': self._format_limit_stats()
            },
            '情绪级别': self._get_sentiment_label(score),
            '术语说明': self._get_glossary()
        }

    def _get_glossary(self) -> Dict:
        """术语说明字典"""
        return {
            '连板数': "指连续涨停天数，例如3连板表示连续3个交易日涨停",
            '非连续涨停': "例如8天5板表示在8个交易日内有5日涨停，但未形成连续涨停",
            '新股过滤': "已排除上市未满60个交易日的股票（涨跌幅规则不同）",
            '有效涨停': "剔除新股、ST股后的真实涨停统计"
        }

    def _format_index_data(self) -> Dict:
        """格式化指数数据为中文"""
        formatted = {}
        for code, data in self._index_data.items():
            formatted[code] = {
                '名称': self._get_index_name(code),
                '当日涨跌幅': f"{data['change_pct']:.2f}%",
                '均线位置': "5日均线上方" if data['position'] == 'above' else "5日均线下方",
                '趋势方向': "上升趋势" if data['trend'] == 'up' else "下降趋势",
                '权重占比': f"{data['weight'] * 100:.1f}%"
            }
        return formatted

    def _format_market_breadth(self) -> Dict:
        """格式化市场广度数据"""
        return {
            '上涨家数': self._market_breadth['rise_num'],
            '下跌家数': self._market_breadth['fall_num'],
            '涨停数量': self._market_breadth['limit_up'],
            '跌停数量': self._market_breadth['limit_down'],
            '涨跌比': f"{self._market_breadth['rise_num'] / self._market_breadth['fall_num']:.2f}:1"
            if self._market_breadth['fall_num'] > 0 else "N/A"
        }

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

    def analyze_premium_effect(self, days=30):
        """
        分析首板次日的溢价效应
        :return: 近期首板次日平均溢价率
        """
        try:
            # 获取历史首板数据
            start_date = self.calendar.get_previous_trade_date(self.trade_date, days)
            zt_df = ak.stock_zt_pool_em(start_date)

            # 筛选首板
            first_zt = zt_df[zt_df['连板数'] == 1]

            # 获取次日开盘价
            premiums = []
            for _, row in first_zt.iterrows():
                next_date = self.calendar.get_next_trade_date(row['日期'])
                if next_date:
                    day_kline = ak.stock_zh_a_hist(symbol=row['代码'], period='daily',
                                                   start_date=next_date, end_date=next_date)
                    if not day_kline.empty:
                        open_pct = (day_kline.iloc[0]['开盘'] / row['收盘价'] - 1) * 100
                        premiums.append(open_pct)

            return sum(premiums) / len(premiums) if premiums else 0

        except Exception as e:
            logger.error(f"溢价分析失败: {str(e)}")
            return 0

# 使用示例
if __name__ == "__main__":

    analyzer = MarketSentimentAnalyzer()
    report = analyzer.generate_report()

    print(f"最高连板数: {report['市场数据']['涨停分析']['最高连板数']}")
    print("连板分布:")
    for k, v in report['市场数据']['涨停分析']['连板分布'].items():
        print(f"  {k}: {v}家")

    if report['市场数据']['涨停分析']['特殊涨停案例']:
        print("\n📌 非连续涨停案例:")
        for case in report['市场数据']['涨停分析']['特殊涨停案例']:
            print(f"  - {case}")

    extreme_data = analyzer.detect_extreme_boards()
    print(f"检测到天地板：{extreme_data['sky_earth']}例，地天板：{extreme_data['earth_sky']}例")
    print(json.dumps(extreme_data['details'], indent=4, ensure_ascii=False))
