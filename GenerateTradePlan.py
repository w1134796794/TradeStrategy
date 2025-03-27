from datetime import datetime
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor
from GetLHB import LHBProcessor
from GetMarketSentiment import MarketSentimentAnalyzer
from GetHotSectors import SectorAnalyzer
from GetTradeDate import TradeCalendar
from Strategies import ScoringStrategy, PositionStrategy, RiskControlStrategy
from typing import List, Tuple, Dict, Set
import akshare as ak

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
            'min_score': int(65),
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

        if self.is_market_hours():
            # 交易时段使用当日数据
            data_date = self.calendar.get_previous_trade_date(current_date)
            plan_date = current_date
        else:
            # 非交易时段
            if current_date in self.calendar.sorted_dates:
                # 当天是交易日，但非交易时段
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
        # 假设交易时段为9:30-15:00
        return (now >= datetime.strptime("09:30", "%H:%M").time() and now <= datetime.strptime("15:00", "%H:%M").time())

    def should_generate_next_day_plan(self):
        """判断是否需要生成次日计划"""
        now = datetime.now()
        current_time = now.time()
        # 条件1：交易时段外（15:00后）
        # 条件2：周末且周五收盘后
        return (
                (not self.is_market_hours()) or
                (now.weekday() >= 5)  # 周六/周日
        )

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

    def _collect_market_data(self, data_date: str) -> Dict:
        """并行收集市场数据（确保返回正确类型）"""
        from concurrent.futures import as_completed

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
            return self._preprocess_zt_data(df)
        except Exception as e:
            logger.error(f"涨停数据获取失败: {str(e)}")
            return pd.DataFrame()

    def _preprocess_zt_data(self, zt_df: pd.DataFrame) -> pd.DataFrame:
        """预处理涨停板数据（修改后）"""
        zt_df = zt_df.rename(columns={
            '代码': 'code',
            '名称': 'name',
            '封板资金': 'seal_amount',
            '首次封板时间': 'first_seal_time'
        })

        # 使用策略类处理连板数和过滤
        filtered = ScoringStrategy.filter_early_boards(zt_df)

        # 强制类型转换（处理异常值）
        filtered['limit_count'] = (
            pd.to_numeric(filtered['limit_count'], errors='coerce')
            .fillna(1)  # 无效值默认1连板
            .astype(int)
        )

        filtered['seal_amount'] = (
            pd.to_numeric(filtered['seal_amount'], errors='coerce')
            .fillna(0.0)
            .astype(float)
        )

        return filtered[['code', 'name', 'limit_count', 'seal_amount', 'first_seal_time']]

    def _generate_candidates(self, data_pack: Dict) -> pd.DataFrame:
        """候选股生成（参数修复版）"""
        try:
            # 确保基础数据存在
            if 'zt_data' not in data_pack or data_pack['zt_data'].empty:
                raise ValueError("涨停数据缺失")

            # ========== 处理龙虎榜数据 ==========
            # 确保有默认值
            lhb_scores = self._calculate_lhb_scores(data_pack.get('lhb_data', pd.DataFrame()))
            merged = data_pack['zt_data'].copy()

            # 添加lhb_score列（带默认值）
            merged['lhb_score'] = merged['code'].map(lhb_scores).fillna(0.0)

            # ========== 类型安全转换 ==========
            type_conversion = {
                'limit_count': {'dtype': int, 'default': 1},
                'seal_amount': {'dtype': float, 'default': 0.0},
                'lhb_score': {'dtype': float, 'default': 0.0}
            }

            for col, config in type_conversion.items():
                merged[col] = (
                    pd.to_numeric(merged[col], errors='coerce')
                    .fillna(config['default'])
                    .astype(config['dtype'])
                )

            # ========== 评分计算 ==========
            # 添加输入验证
            required_columns = ['limit_count', 'seal_amount', 'lhb_score']
            if not all(col in merged.columns for col in required_columns):
                missing = [col for col in required_columns if col not in merged.columns]
                raise KeyError(f"缺失关键列: {missing}")

            merged['total_score'] = merged.apply(
                lambda x: float(ScoringStrategy.calculate_total_score(x)),
                axis=1
            ).clip(0, 100)

            # ========== 最终过滤 ==========
            # 确保参数类型正确
            min_score = int(self.params['min_score'])
            if not pd.api.types.is_numeric_dtype(merged['total_score']):
                raise TypeError("评分列必须为数值类型")

            return (
                merged[merged['total_score'] >= min_score]
                .sort_values('total_score', ascending=False)
                .head(self.params['max_candidates'])
            )

        except Exception as e:
            logger.error(
                f"候选股生成失败（详细日志）: {str(e)}\n当前列: {merged.columns.tolist() if 'merged' in locals() else '无数据'}")
            return pd.DataFrame()

    def _calculate_lhb_scores(self, lhb_data: pd.DataFrame) -> Dict:
        """计算龙虎榜评分（增强版）"""
        if lhb_data.empty:
            return {}

        try:
            # 强制类型转换
            lhb_data['买入金额'] = pd.to_numeric(lhb_data['买入金额'], errors='coerce').fillna(0)
            lhb_data['卖出金额'] = pd.to_numeric(lhb_data['卖出金额'], errors='coerce').fillna(0)

            scores = (
                lhb_data.groupby('代码')
                .apply(lambda x: (x['买入金额'].sum() / (x['卖出金额'].sum() + 1e-6)) * 100)
                .to_dict()
            )
            return {k: min(float(v), 100) for k, v in scores.items()}

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
        if lhb_data.empty:
            return 'unknown'

        buy_ratio = lhb_data['买入金额'].sum() / (lhb_data['卖出金额'].sum() + 1e6)
        return 'good' if buy_ratio > 1.2 else 'normal' if buy_ratio > 0.8 else 'poor'

    def _format_candidates(self, candidates: pd.DataFrame, market_status: Dict) -> List[Dict]:
        """格式化候选股信息"""
        return [{
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
        } for _, row in candidates.iterrows()]

    def _get_tech_reason(self, row: pd.Series) -> str:
        """生成技术面理由"""
        parts = []
        if row['limit_count'] >= 3:
            parts.append(f"{row['limit_count']}连板强势")
        if pd.to_datetime(row['first_seal_time']).hour < 10:
            parts.append("早盘快速涨停")
        if row['seal_amount'] > 1e8:
            parts.append(f"封单{row['seal_amount'] / 1e8:.1f}亿")
        return "，".join(parts) if parts else "技术形态突破"

    def _get_capital_reason(self, row: pd.Series) -> str:
        """生成资金面理由"""
        parts = []
        if row['lhb_score'] > 70:
            parts.append("龙虎榜资金青睐")
        if row.get('has_institution'):
            parts.append("机构席位参与")
        return "，".join(parts) if parts else "资金持续流入"

    def _get_sector_reason(self, row: pd.Series) -> str:
        """生成板块理由"""
        if not isinstance(row['hot_sectors'], list):
            return ""

        top_sectors = sorted(row['hot_sectors'], key=lambda x: x[1], reverse=True)[:2]
        return " + ".join([f"{s[0]}({s[1]}分)" for s in top_sectors])

    def _analyze_sectors(self, sectors: List) -> List[Dict]:
        """生成板块分析报告"""
        return [{
            'name': s[0],
            'type': s[1],
            'momentum': s[2],
            'leader': self._get_sector_leader(s[0])
        } for s in sectors[:5]]

    def _get_sector_leader(self, sector: str) -> str:
        """获取板块龙头股"""
        # 此处需要接入板块成分股数据
        return "待实现"

    def _extract_lhb_insights(self, lhb_data: pd.DataFrame) -> Dict:
        """提取龙虎榜洞见"""
        if lhb_data.empty:
            return {}

        return {
            'top_buyers': lhb_data.groupby('营业部名称')['买入金额'].sum().nlargest(3).to_dict(),
            'institution_net': lhb_data[lhb_data['营业部名称'].str.contains('机构')]
            .apply(lambda x: x['买入金额'].sum() - x['卖出金额'].sum(), axis=1).sum()
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