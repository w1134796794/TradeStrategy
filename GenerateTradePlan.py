from datetime import datetime
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from GetLHB import LHBProcessor
from GetMarketSentiment import MarketSentimentAnalyzer
from GetHotSectors import SectorAnalyzer
from GetTradeDate import TradeCalendar
from Strategies import ScoringStrategy, PositionStrategy, RiskControlStrategy, FilterStrategy
from typing import List, Dict
import akshare as ak
from Cache_Manager import default_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradePlanGenerator:
    """交易计划生成器"""

    def __init__(self):
        self.calendar = TradeCalendar()
        self.market_analyzer = MarketSentimentAnalyzer()
        self.sector_analyzer = SectorAnalyzer()
        self.lhb_processor = LHBProcessor()
        self.params = {
            'max_candidates': int(8),
            'min_score': int(45),
            'sector_threshold': int(3)
        }

    def generate_daily_plan(self) -> Dict:
        """生成完整交易计划"""
        plan_date = datetime.now().strftime("%Y%m%d")
        try:
            # 步骤1：获取交易日期
            data_date, plan_date = self._get_trade_dates()

            # 步骤2：市场分析
            market_status = self._analyze_market(data_date)
            if not market_status.get('tradeable', False):
                return self._generate_empty_plan(plan_date, market_status.get('message', "暂停交易"))

            # 步骤3：数据收集
            data_pack = self._collect_market_data(data_date)
            if data_pack['zt_data'].empty:
                return self._generate_empty_plan(plan_date, "无有效涨停数据")

            # 步骤4-5
            # 修复：移除多余的 market_status 参数
            candidates = self._generate_candidates(data_pack)
            return self._compile_full_plan(plan_date, candidates, market_status, data_pack)

        except Exception as e:
            logger.error(f"计划生成失败: {str(e)}", exc_info=True)
            return self._generate_empty_plan(plan_date, f"系统异常: {str(e)}")

    def _get_trade_dates(self) -> tuple:
        """获取数据日期和计划日期"""
        now = datetime.now()
        current_date = now.strftime("%Y%m%d")
        current_time = now.time()

        if self.is_market_hours():
            # 交易时段使用当日数据
            data_date = self.calendar.get_previous_trade_date(current_date)
            plan_date = current_date
        else:
            if current_date in self.calendar.sorted_dates:
                if current_time >= datetime.strptime("15:00", "%H:%M").time():
                    # 当天是交易日，且时间在 15:00 之后
                    data_date = current_date
                    plan_date = self.calendar.get_next_trade_date(current_date)
                else:
                    # 当天是交易日，但时间在 15:00 之前
                    data_date = self.calendar.get_previous_trade_date(current_date)
                    plan_date = current_date
            else:
                # 当天非交易日
                plan_date = self.calendar.get_next_trade_date()
                data_date = self.calendar.get_previous_trade_date(plan_date)

        logger.info(f"日期获取结果 | 当前时间:{now} 数据日期:{data_date} 计划日期:{plan_date}")
        return data_date, plan_date

    def is_market_hours(self):
        """判断当前是否在交易时段内（9:30-15:00）"""
        now = datetime.now().time()
        # 假设交易时段为 9:30-15:00
        return (now >= datetime.strptime("09:30", "%H:%M").time() and now <= datetime.strptime("15:00", "%H:%M").time())

    def _analyze_market(self, data_date: str) -> dict:
        """健壮的市场分析方法"""
        try:
            # 确保数据正确加载
            self.market_analyzer.collect_market_data()

            # 添加数据有效性检查
            if not hasattr(self.market_analyzer, '_market_breadth'):
                raise ValueError("市场广度数据未正确加载")

            score = self.market_analyzer.calculate_total_score()
            breadth = self.market_analyzer._market_breadth

            return {
                'score': score,
                'level': self.market_analyzer.get_sentiment_label(score),
                'message': "",
                'tradeable': score >= 60,
                'limit_up': breadth.get('limit_up', 0),
                'limit_down': breadth.get('limit_down', 0)
            }
        except Exception as e:
            logger.error(f"市场分析异常: {str(e)}")
            return {
                'score': 0,
                'level': "异常",
                'message': str(e),
                'tradeable': False,
                'limit_up': 0,
                'limit_down': 0
            }

    @default_cache(prefix="market_cache", ttl=7200)
    def _collect_market_data(self, data_date: str) -> Dict:
        """并行收集市场数据（确保返回正确类型）"""

        results = {
            'zt_data': pd.DataFrame(),
            'lhb_data': pd.DataFrame(),
            'sectors': []
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._get_zt_data, data_date): 'zt_data',
                executor.submit(self.lhb_processor.get_enhanced_data, [data_date]): 'lhb_data',
                executor.submit(self.sector_analyzer.get_hot_sectors): 'sectors'
            }

            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                    # 类型转换保证
                    if key == 'sectors' and isinstance(result, pd.DataFrame):
                        results[key] = result.to_dict('records')
                    elif key != 'sectors' and not isinstance(result, pd.DataFrame):
                        results[key] = pd.DataFrame(result)
                    else:
                        results[key] = result
                except Exception as e:
                    logger.error(f"{key}数据采集失败: {str(e)}")

        return results

    def _get_zt_data(self, date: str) -> pd.DataFrame:
        """获取涨停板数据并进行预处理"""
        try:
            df = ak.stock_zt_pool_em(date=date)
            return self._preprocess_zt_data(df, date)
        except Exception as e:
            logger.error(f"涨停数据获取失败: {str(e)}")
            return pd.DataFrame()

    def _preprocess_zt_data(self, zt_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """预处理涨停板数据（修复时间解析错误）"""
        # 重命名字段
        zt_df = zt_df.rename(columns={
            '代码': 'code',
            '名称': 'name',
            '最新价': 'close',
            '封板资金': 'seal_amount',
            '首次封板时间': 'first_seal_time',
            '连板数': 'limit_count',
            '流通市值': 'circ_market_cap'
        })

        # 解析时间并合并日期
        def parse_datetime(row):
            try:
                # 1. 格式化日期部分
                formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"  # YYYY-MM-DD

                # 2. 处理时间字符串
                raw_time = str(row['first_seal_time'])
                time_str = raw_time.zfill(6)  # 补全为6位

                # 3. 拆分时、分、秒并验证合法性
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])

                if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
                    raise ValueError(f"非法时间值: {time_str}")

                # 4. 拼接完整日期时间
                datetime_str = f"{formatted_date} {hour:02}:{minute:02}:{second:02}"

                # 5. 解析为datetime对象
                return pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S')
            except Exception as e:
                logger.error(f"解析失败 | 原始值: {raw_time} | 错误: {str(e)}")
                return pd.NaT

        # 应用解析逻辑
        zt_df['datetime'] = zt_df.apply(parse_datetime, axis=1)

        # 过滤无效数据
        zt_df = zt_df.dropna(subset=['datetime'])

        # 强制类型转换
        zt_df['limit_count'] = (
            pd.to_numeric(zt_df['limit_count'], errors='coerce')
            .fillna(1)
            .astype(int)
        )
        zt_df['seal_amount'] = (
            pd.to_numeric(zt_df['seal_amount'], errors='coerce')
            .fillna(0.0)
            .astype(float)
        )
        zt_df['circ_market_cap'] = (
            pd.to_numeric(zt_df['circ_market_cap'], errors='coerce')
            .fillna(0.0)
            .astype(float)
        )

        # 返回所有必要列
        return zt_df[['code', 'name', 'close', 'limit_count', 'seal_amount', 'datetime', 'first_seal_time', 'circ_market_cap']]

    def _generate_candidates(self, data_pack: Dict) -> pd.DataFrame:
        """候选股生成（参数修复版）"""
        try:
            # 1. 基础数据校验
            zt_df = data_pack.get('zt_data', pd.DataFrame())
            if zt_df.empty:
                raise ValueError("涨停数据为空")

            lhb_data = data_pack.get('lhb_data', pd.DataFrame())

            # 计算龙虎榜评分并合并到涨停数据
            if not lhb_data.empty:
                lhb_scores = self._calculate_lhb_scores(lhb_data)
                zt_df['lhb_score'] = zt_df['code'].map(lhb_scores).fillna(0)
            else:
                zt_df['lhb_score'] = 0  # 无数据时填充默认值

            # 2. 过滤首板/二板
            filtered = FilterStrategy.filter_early_boards(zt_df)

            # 获取热点板块成分股映射
            hot_sectors = data_pack.get('sectors', [])
            sector_stocks_map = self.sector_analyzer.build_sector_map(hot_sectors)

            # 构建板块名到板块涨幅的映射
            sector_score_map = {
                (name, sector_type): score
                for name, sector_type, score in hot_sectors
            }

            # 为每只股票添加所属板块及板块涨幅
            def map_sectors(code: str) -> list:
                matched = []
                for sector_key in sector_stocks_map:
                    try:
                        name, stype = sector_key.split('_', 1)
                    except ValueError:
                        continue
                    if code in sector_stocks_map[sector_key]:
                        score = sector_score_map.get((name, stype), 0)
                        matched.append((name, score))
                return matched

            filtered['hot_sectors'] = filtered['code'].apply(map_sectors)

            # 4. 计算综合得分（关键修复：传递 sector_heat_map）
            filtered['total_score'] = filtered.apply(
                lambda row: ScoringStrategy.calculate_total_score(row, sector_stocks_map),
                axis=1
            )

            # 5. 按得分排序并返回 Top N
            return (
                filtered[filtered['total_score'] >= self.params['min_score']]
                .sort_values('total_score', ascending=False)
                .head(self.params['max_candidates'])
            )

        except Exception as e:
            logger.error(f"候选股生成失败: {str(e)}")
            return pd.DataFrame()

    def _calculate_lhb_scores(self, lhb_data: pd.DataFrame) -> Dict:
        """计算龙虎榜评分（增强版）"""
        if lhb_data.empty:
            return {}

        try:
            # 强制类型转换
            lhb_data['买入金额'] = pd.to_numeric(lhb_data['买入金额'], errors='coerce').fillna(0)
            lhb_data['卖出金额'] = pd.to_numeric(lhb_data['卖出金额'], errors='coerce').fillna(0)

            scores = {}
            for code, group in lhb_data.groupby('代码'):
                buy_sum = group['买入金额'].sum()
                sell_sum = group['卖出金额'].sum()
                score = (buy_sum / (sell_sum + 1e-6)) * 100
                scores[code] = min(float(score), 100.0)
            return scores

        except Exception as e:
            logger.error(f"龙虎榜评分计算失败: {str(e)}")
            return {}

    def _compile_full_plan(self, plan_date: str, candidates: pd.DataFrame,
                           market_status: Dict, data_pack: Dict) -> Dict:
        """编译完整交易计划（确保所有字段存在）"""

        plan = {
            'plan_date': plan_date,
            'market_status': self._format_market(market_status),
            'risk_assessment': {
                'market_risk': 'unknown',
                'sector_risk': 'unknown',
                'liquidity_risk': 'unknown'
            },
            'candidates': [],
            'sector_analysis': [],
            'lhb_insights': {}
        }

        try:
            # 填充风险数据
            plan['risk_assessment'] = {
                'market_risk': RiskControlStrategy.check_market_risk(market_status),
                'sector_risk': self._check_sector_risk(data_pack.get('sectors', [])),
                'liquidity_risk': self._check_liquidity(data_pack.get('lhb_data', pd.DataFrame()))
            }

            # 填充候选股数据
            if not candidates.empty:
                plan['candidates'] = self._format_candidates(candidates, market_status)
            # 填充板块分析
            plan['sector_analysis'] = self._analyze_sectors(data_pack.get('sectors', []))
            # 填充龙虎榜数据
            if 'lhb_data' in data_pack and not data_pack['lhb_data'].empty:
                plan['lhb_insights'] = self._extract_lhb_insights(data_pack['lhb_data'])
        except Exception as e:
            logger.error(f"计划编译异常: {str(e)}")

        return plan

    def _format_market(self, status: Dict) -> Dict:
        """格式化市场状态数据"""
        return {
            'score': status['score'],
            'level': status['level'],
            'limit_up': status.get('limit_up', 0),
            'limit_down': status.get('limit_down', 0),
            'max_limit': status.get('max_limit', 0)
        }

    def _assess_risks(self, market_status: Dict, data_pack: Dict) -> Dict:
        """综合风险评估"""
        return {
            'market_risk': RiskControlStrategy.check_market_risk(market_status),
            'sector_risk': self._check_sector_risk(data_pack['sectors']),
            'liquidity_risk': self._check_liquidity(data_pack['lhb_data'])
        }

    def _check_sector_risk(self, sectors: List) -> str:
        """板块过热风险评估"""
        hot_count = sum(1 for s in sectors if s[2] > 80)  # s[2] 对应得分
        return 'high' if hot_count > 3 else 'medium' if hot_count > 1 else 'low'

    def _check_liquidity(self, lhb_data: pd.DataFrame) -> str:
        """流动性风险评估"""
        if lhb_data.empty or '买入金额' not in lhb_data.columns or '卖出金额' not in lhb_data.columns:
            return 'unknown'
        try:
            # 强制类型转换并求和
            buy = pd.to_numeric(lhb_data['买入金额'], errors='coerce').fillna(0).sum()
            sell = pd.to_numeric(lhb_data['卖出金额'], errors='coerce').fillna(0).sum()
            buy_ratio = buy / (sell + 1e-6)
            return 'good' if buy_ratio > 1.2 else 'normal' if buy_ratio > 0.8 else 'poor'
        except Exception as e:
            logger.error(f"流动性评估异常: {str(e)}")
            return 'unknown'

    def _format_candidates(self, candidates: pd.DataFrame, market_status: Dict) -> List[Dict]:
        """格式化候选股信息（修复时间格式化问题）"""
        formatted = []

        candidates['first_seal_time'] = candidates['first_seal_time'].astype(str).apply(
            lambda x: f"{x[:2]}:{x[2:4]}:{x[4:]}" if len(x) == 6 else x  # 142612 → 14:26:12
        )
        candidates['datetime'] = pd.to_datetime(candidates['datetime'])  # 确保日期列是datetime类型

        for idx, row in candidates.iterrows():
            try:
                formatted.append({
                    'code': row['code'],
                    'name': row['name'],
                    'score': row['total_score'],
                    'position': PositionStrategy.calculate_position(row, market_status),
                    'entry_price': row['close'] * 1.03,
                    'stop_loss': row['close'] * 0.95,
                    'reasons': {
                        'technical': self._get_tech_reason(row),
                        'capital': self._get_capital_reason(row),
                        'sector': self._get_sector_reason(row)
                    }
                })
            except Exception as e:
                logger.error(
                    f"格式化候选股失败: {row['code']} | 错误: {str(e)} | 关键值: "
                    f"first_seal_time={row['first_seal_time']}, datetime={row['datetime']}"
                )
        return formatted

    def _get_tech_reason(self, row: pd.Series) -> str:
        """生成技术面理由（封单占比版）"""
        parts = []

        # 连板数判断
        if row['limit_count'] >= 3:
            parts.append(f"{row['limit_count']}连板强势")

        # 早盘涨停判断
        if pd.to_datetime(row['first_seal_time']).hour < 10:
            parts.append("早盘快速涨停")

        # 封单占比判断（新增逻辑）
        if row.get('circ_market_cap', 0) > 0:
            seal_ratio = row['seal_amount'] / row['circ_market_cap']
            if seal_ratio > 0.015:
                parts.append(f"封单占流通市值{seal_ratio * 100:.1f}%")

        return "，".join(parts) if parts else "技术形态突破"

    def _get_capital_reason(self, row: pd.Series) -> str:
        """生成资金面理由"""
        parts = []
        # 使用 get 方法避免 KeyError，并设置默认值
        lhb_score = row.get('lhb_score', 0)
        if lhb_score > 70:
            parts.append("龙虎榜资金青睐")

        # 检查是否存在机构席位字段
        has_institution = row.get('has_institution', False)
        if has_institution:
            parts.append("机构席位参与")

        return "，".join(parts) if parts else "资金持续流入"

    def _get_sector_reason(self, row: pd.Series) -> str:
        """生成板块理由"""
        try:
            # 处理可能的缺失列或空值
            print(f"row:\n{row}")
            sectors = row.get('hot_sectors', [])
            print(f"sectors:\n{sectors}")
            if not isinstance(sectors, list) or len(sectors) == 0:
                return ""

            top_sectors = sorted(sectors, key=lambda x: x[1], reverse=True)[:2]
            return " + ".join([f"{name}(涨幅: {score}%)" for name, score in top_sectors])
        except Exception as e:
            logger.error(f"生成板块理由失败: {str(e)}")
            return ""

    def _analyze_sectors(self, sectors: List) -> List[Dict]:
        """生成板块分析报告"""
        try:
            return [{
                'name': sector_info[0],
                'type': sector_info[1],
                'momentum': sector_info[2],
                'leader': self.sector_analyzer.get_sector_dragons(
                    sector=sector_info[0],
                    sector_type=sector_info[1]
                )
            } for sector_info in sectors[:5]]
        except IndexError as e:
            logger.error(f"板块数据结构异常: {str(e)}")
            return []

    def _extract_lhb_insights(self, lhb_data: pd.DataFrame) -> Dict:
        """提取龙虎榜洞见"""
        if lhb_data.empty:
            return {}

            # 修正机构净买入计算逻辑
        institution_df = lhb_data[lhb_data['营业部名称'].str.contains('机构')]
        institution_net = (
                institution_df['买入金额'].sum() -
                institution_df['卖出金额'].sum()
        )

        return {
            'top_buyers': lhb_data.groupby('营业部名称')['买入金额'].sum().nlargest(3).to_dict(),
            'institution_net': institution_net
        }

    def _generate_empty_plan(self, plan_date: str, message: str) -> Dict:
        """生成安全的空计划模板"""
        return {
            'plan_date': plan_date,
            'market_status': {
                'score': 0,
                'level': '异常',
                'message': message,
                'tradeable': False
            },
            'risk_assessment': {
                'market_risk': 'unknown',
                'sector_risk': 'unknown',
                'liquidity_risk': 'unknown'
            },
            'candidates': [],
            'sector_analysis': [],
            'lhb_insights': {}
        }