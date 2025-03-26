from datetime import datetime
import pandas as pd
import akshare as ak
import logging
from concurrent.futures import ThreadPoolExecutor
from LHB import LHBProcessor
from MarketSentiment import MarketSentimentConfig, MarketSentimentAnalyzer
from HotSectors import SectorAnalyzer
from TradeDate import TradeCalendar
import re


# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSystem:
    """交易系统主控类"""

    def __init__(self):
        # 初始化各个模块
        self.calendar = TradeCalendar()
        self.market_analyzer = MarketSentimentAnalyzer()
        self.sector_analyzer = SectorAnalyzer()
        self.lhb_processor = LHBProcessor()
        self.show_score_details = True  # 控制是否显示得分明细

        # 配置参数
        self.params = {
            'market_sentiment_threshold': 60,  # 市场情绪合格线
            'max_candidates': 5,  # 最大候选股数量
            'position_limits': {  # 仓位限制规则
                'default': 20,
                'high_score': 30,
                'sector_leader': 35
            }
        }
        self.period = "近一月"

    def _get_trade_dates(self):
        """获取双日期逻辑：数据日期和计划日期"""
        if self.calendar.should_generate_next_day_plan():
            plan_date = self.calendar.get_next_trade_date()
            data_date = self.calendar.get_previous_trade_date(plan_date)
        else:
            data_date = self.calendar.get_recent_trade_date()
            plan_date = self.calendar.get_next_trade_date(data_date)

        # 边界检查
        if not data_date or not plan_date:
            raise ValueError("日期计算异常")
        self.data_date = data_date
        return data_date, plan_date


    def run(self):
        """执行完整交易流程"""
        try:
            # 步骤1：获取双日期
            data_date, plan_date = self._get_trade_dates()
            logger.info(f"数据日期: {data_date} | 计划日期: {plan_date}")

            # 步骤2：市场情绪分析（基于数据日期）
            self.market_analyzer.trade_date = data_date  # 需在MarketSentimentAnalyzer中添加trade_date属性

            # 步骤2：市场情绪分析
            market_status = self._analyze_market(data_date)
            if not market_status['tradeable']:
                return self._generate_empty_plan(data_date, market_status['message'])

            # 步骤3：获取核心数据
            data_pack = self._collect_data(data_date)
            if data_pack['zt_df'].empty:
                return self._generate_empty_plan(data_date, "无有效涨停数据")

            # 步骤4：生成候选股
            candidates = self._generate_candidates(data_pack)

            # 步骤5：生成最终交易计划
            return self._generate_full_report(plan_date, candidates, market_status, data_pack)

        except Exception as e:
            logger.error(f"系统运行异常: {str(e)}", exc_info=True)
            return self._generate_empty_plan(plan_date, "系统异常")

    def _generate_full_report(self, plan_date, candidates, market_status, data_pack):
        """生成包含完整市场数据的报告"""
        report = {
            "交易日期": plan_date,
            "市场情绪": self._format_market_sentiment(market_status),
            "热门板块": self._format_hot_sectors(),
            "龙虎榜数据": self._format_lhb_data(data_pack['lhb_data']),
            "涨停分析": self._format_zt_analysis(data_pack['zt_df']),
            "候选股票": self._format_candidates(candidates, market_status),  # 添加market_status参数
            "风险提示": self._generate_risk_notes(market_status['score'])
        }
        self._print_report(report)
        return report

    def _format_market_sentiment(self, market_status):
        """格式化市场情绪数据"""
        analyzer = self.market_analyzer
        return {
            "综合评分": f"{market_status['score']}/100",
            "情绪级别": analyzer._get_sentiment_label(market_status['score']),
            "关键指标": {
                "涨停数量": analyzer._market_breadth.get('limit_up', 0),
                "跌停数量": analyzer._market_breadth.get('limit_down', 0),
                "上涨家数": analyzer._market_breadth.get('rise_num', 0),
                "下跌家数": analyzer._market_breadth.get('fall_num', 0)
            },
            "连板数据": {
                "最高连板": analyzer._limit_stats.get('max_limit', 0),
                "连板分布": analyzer._limit_stats.get('distribution', {})
            },
            "极端波动": self._format_extreme_cases(analyzer.detect_extreme_boards())
        }

    def _format_extreme_cases(self, extreme_data):
        """格式化极端案例"""
        return {
            "天地板数量": extreme_data['sky_earth'],
            "地天板数量": extreme_data['earth_sky'],
            "典型案例": [
                f"{d['name']}({d['code']}) {d['type']} 振幅{d['amplitude']:.2%}"
                for d in extreme_data['details'][:3]  # 显示前3个案例
            ]
        }

    def _format_hot_sectors(self):
        """格式化板块数据"""
        sectors = self.sector_analyzer.hot_sectors
        return [
            {
                "板块名称": sector[0],
                "板块类型": "行业" if sector[1] == "industry" else "概念",
                "动量评分": self.sector_analyzer.get_sector_momentum(*sector),
                "成分股数量": len(self.sector_analyzer.sector_stocks_map.get(sector, []))
            }
            for sector in sectors
        ]

    def _format_lhb_data(self, lhb_data):
        """格式化龙虎榜数据"""
        if lhb_data.empty:
            return "当日无龙虎榜数据"

        top3 = lhb_data.sort_values('买入金额', ascending=False).head(3)
        return {
            "活跃营业部": [
                f"{row['营业部名称']} 买入{row['买入金额'] / 1e8:.2f}亿"
                for _, row in top3.iterrows()
            ],
            "机构动向": {
                "机构买入总额": lhb_data[lhb_data['营业部名称'].str.contains('机构专用')]['买入金额'].sum() / 1e8,
                "游资代表": lhb_data[
                    lhb_data['营业部名称'].str.contains('华鑫|中信|东财')
                ]['营业部名称'].unique().tolist()
            }
        }

    def _format_zt_analysis(self, zt_df):
        """涨停板分析"""
        return {
            "涨停总数": len(zt_df),
            "连板分布": zt_df['连板数'].value_counts().to_dict(),
            "强势特征": {
                "早盘涨停(10点前)": zt_df[zt_df['首次封板时间'] < '10:00'].shape[0],
                "高封单金额(>1亿)": zt_df[zt_df['封板资金'] > 1e8].shape[0]
            }
        }

    def _format_candidates(self, candidates, market_status):
        """格式化候选股数据"""
        return [
            {
                "代码": row['代码'],
                "名称": row['名称'],
                "推荐理由": {
                    "技术面": [
                        f"{row['连板数']}连板",
                        f"封板时间: {row['首次封板时间']}",
                        f"封单金额: {row['封板资金'] / 1e8:.2f}亿"
                    ],
                    "资金面": [
                        f"龙虎榜评分: {row.get('lhb_score', 0)}",
                        f"机构买入: {'有' if row.get('has_institution') else '无'}"
                    ],
                    "题材面": [
                        f"{s[0]}({self.sector_analyzer.get_sector_momentum(*s)}分)"
                        for s in row['hot_sectors'][:2]
                    ]
                },
                "操作建议": {
                    "建议仓位": f"{self._calculate_position(row, market_status['score'])}%",
                    "入场价格": f"不超过{row['最新价'] * 1.03:.2f}元",
                    "止损策略": f"跌破{row['最新价'] * 0.95:.2f}元止损"
                }
            }
            for _, row in candidates.iterrows()
        ]

    def _generate_risk_notes(self, market_score):
        """生成风险提示"""
        notes = []
        if market_score < 40:
            notes.append("⚠️ 市场情绪低迷，建议仓位控制在30%以下")
        if self.market_analyzer._limit_stats.get('max_limit', 0) >= 7:
            notes.append("🚨 市场出现高度连板股，注意分化风险")
        if self.market_analyzer._market_breadth.get('limit_down', 0) > 10:
            notes.append("‼️ 跌停数量超过10家，注意系统性风险")
        return notes or ["市场风险在正常范围内"]

    def _print_report(self, report):
        """控制台美化输出"""
        print(f"\n{'=' * 30} 交易日报 【{report['交易日期']}】 {'=' * 30}")

        # 市场情绪板块
        print("\n🔔 市场情绪分析：")
        senti = report['市场情绪']
        print(f"| 综合评分: {senti['综合评分']} | 情绪级别: {senti['情绪级别']} |")
        print("| 涨停/跌停: {}/{} | 涨跌比: {}/{} |".format(
            senti['关键指标']['涨停数量'], senti['关键指标']['跌停数量'],
            senti['关键指标']['上涨家数'], senti['关键指标']['下跌家数']
        ))
        print(f"| 最高连板: {senti['连板数据']['最高连板']} | 连板分布: {senti['连板数据']['连板分布']} |")
        print(f"| 极端波动: 天地板{senti['极端波动']['天地板数量']}例 地天板{senti['极端波动']['地天板数量']}例 |")

        # 热门板块
        print("\n🔥 热门板块追踪：")
        for sector in report['热门板块']:
            print(
                f"| {sector['板块名称']: <10} | 类型: {sector['板块类型']: <5} | 动量: {sector['动量评分']: <3}分 | 成分股: {sector['成分股数量']: <3}只 |")

        # 龙虎榜数据
        print("\n💰 龙虎榜焦点：")
        lhb = report['龙虎榜数据']
        if isinstance(lhb, str):
            print(lhb)
        else:
            print("活跃席位：", " | ".join(lhb['活跃营业部'][:3]))
            print(f"机构动向：净买入{lhb['机构动向']['机构买入总额']:.2f}亿")
            print("知名游资：", "、".join(lhb['机构动向']['游资代表'][:3]))

        # 候选股票
        print("\n🚀 精选候选股：")
        for stock in report['候选股票']:
            print(f"\n► {stock['名称']}({stock['代码']})")
            print(f"| 技术面: {' | '.join(stock['推荐理由']['技术面'])}")
            print(f"| 资金面: {' | '.join(stock['推荐理由']['资金面'])}")
            print(f"| 题材面: {' | '.join(stock['推荐理由']['题材面'])}")
            print(
                f"📈 操作建议: {stock['操作建议']['建议仓位']} | {stock['操作建议']['入场价格']} | {stock['操作建议']['止损策略']}")

        # 风险提示
        print("\n⚠️ 风险提示：")
        for note in report['风险提示']:
            print(f"| {note}")

        print("\n" + "=" * 70)
    def _analyze_market(self, trade_date):
        """执行市场分析"""
        market_data = self.market_analyzer.collect_market_data()
        score = self.market_analyzer.calculate_total_score()

        return {
            'score': score,
            'tradeable': score >= self.params['market_sentiment_threshold'],
            'message': f"市场情绪分数 {score} 低于阈值 {self.params['market_sentiment_threshold']}" if score < self.params['market_sentiment_threshold'] else ""
        }

    def _collect_data(self, trade_date):
        """支持获取历史数据"""
        try:
            # 尝试获取当天数据
            zt_df = ak.stock_zt_pool_em(date=trade_date)
        except Exception as e:
            logger.warning(f"当日数据获取失败，使用最近历史数据: {str(e)}")
            # 获取前一个交易日的缓存数据
            prev_date = self.calendar.get_previous_trade_date(trade_date)
            zt_df = ak.stock_zt_pool_em(date=prev_date)

        with ThreadPoolExecutor() as executor:
            # 并行获取数据
            sector_future = executor.submit(self.sector_analyzer.get_hot_sectors)
            lhb_future = executor.submit(
                self.lhb_processor.get_enhanced_data,
                dates=[trade_date],  # 日期列表
                statistic_period=self.period  # 统计周期参数
            )

            # 获取涨停板数据
            zt_df = ak.stock_zt_pool_em(date=trade_date)
            zt_df = self._preprocess_zt_data(zt_df)

            return {
                'zt_df': zt_df,
                'sectors': sector_future.result(),
                'lhb_data': lhb_future.result()
            }

    # def _preprocess_zt_data(self, zt_df):
    #     """预处理涨停板数据"""
    #     required_columns = ['代码', '名称', '最新价', '涨停统计', '封板资金', '首次封板时间']
    #     missing_cols = [col for col in required_columns if col not in zt_df.columns]
    #     if missing_cols:
    #         logger.warning(f"缺少必要字段: {missing_cols}，尝试填充默认值")
    #         for col in missing_cols:
    #             zt_df[col] = 0 if col in ['封板资金'] else ''
    #
    #     # 统一代码格式
    #     zt_df['代码'] = zt_df['代码'].astype(str).str.zfill(6)
    #
    #     # 解析连板数
    #     def parse_limit_count(stat):
    #         try:
    #             if pd.isna(stat):
    #                 return 1  # 默认首板
    #             parts = str(stat).split('/')
    #             return int(parts[1]) if len(parts) > 1 else 1
    #         except:
    #             return 1
    #
    #     zt_df['limit_count'] = zt_df['涨停统计'].apply(parse_limit_count)
    #
    #     # 过滤首板/二板
    #     zt_df = zt_df[zt_df['limit_count'].isin([1, 2])]
    #
    #     # 填充缺失值
    #     zt_df['封板资金'] = zt_df['封板资金'].fillna(0).astype(float)
    #     zt_df['首次封板时间'] = zt_df['首次封板时间'].fillna('15:00')
    #
    #     return zt_df

        # # 解析涨停统计字段
        # if '涨停统计' in zt_df.columns:
        #     split_df = zt_df['涨停统计'].str.split('/', expand=True)
        #     if split_df.shape[1] >= 2:
        #         zt_df['limit_count'] = split_df[1].fillna(0).astype(int)
        #     else:
        #         zt_df['limit_count'] = 0
        #
        # # 过滤无效数据
        # required_cols = ['最新价', '涨停统计', '封板资金', '首次封板时间']
        # return zt_df.dropna(subset=required_cols)

    def parse_limit_count(stat):
        try:
            if pd.isna(stat):
                return 1
            # 匹配"X板"格式
            match = re.search(r'(\d+)板', str(stat))
            return int(match.group(1)) if match else 1
        except:
            return 1

    def _preprocess_zt_data(self, zt_df):
        """改进版涨停板预处理（修复链式赋值问题）"""
        # 创建独立副本以避免视图问题
        zt_df = zt_df.copy()

        # 确保基础字段存在
        required_columns = ['代码', '名称', '最新价', '涨停统计', '封板资金', '首次封板时间']
        missing_cols = [col for col in required_columns if col not in zt_df.columns]

        # 填充缺失列
        for col in missing_cols:
            if col == '封板资金':
                zt_df[col] = 0.0
            elif col == '首次封板时间':
                zt_df[col] = '15:00'
            else:
                zt_df[col] = ''

        # 统一代码格式
        zt_df.loc[:, '代码'] = zt_df['代码'].astype(str).str.zfill(6)

        # 解析连板数（显式使用.loc）
        def parse_limit_count(stat):
            try:
                if pd.isna(stat):
                    return 1  # 默认首板
                return int(str(stat).split('/')[1])
            except:
                return 1

        zt_df.loc[:, 'limit_count'] = zt_df['涨停统计'].apply(parse_limit_count)

        # 过滤首板/二板（创建新副本）
        filtered_df = zt_df.loc[zt_df['limit_count'].isin([1, 2])].copy()

        # 填充缺失值（使用.loc明确赋值）
        filtered_df.loc[:, '封板资金'] = filtered_df['封板资金'].fillna(0).astype(float)
        filtered_df.loc[:, '首次封板时间'] = filtered_df['首次封板时间'].fillna('15:00')

        logger.debug(f"连板数解析示例（前5条）:")
        for idx, row in zt_df[['涨停统计', 'limit_count']].head().iterrows():
            logger.debug(f"{row['涨停统计']} => {row['limit_count']}")

        return filtered_df

    def _generate_candidates(self, data_pack):
        """生成候选股列表"""
        # 关联板块数据
        sector_map = self.sector_analyzer.build_sector_map(
            hot_sectors=data_pack['sectors'],
            max_workers=5  # 明确指定线程数
        )
        zt_df = self._add_sector_info(data_pack['zt_df'], sector_map)

        # 关联龙虎榜数据
        lhb_scores = self._calculate_lhb_scores(data_pack['lhb_data'])
        zt_df['lhb_score'] = zt_df['代码'].map(lhb_scores)

        zt_df['total_score'] = zt_df.apply(self._calculate_total_score, axis=1)

        # 确保每个值都是包含两个元素的元组
        zt_df['total_score'] = zt_df['total_score'].apply(
            lambda x: (x[0], x[1]) if isinstance(x, tuple) else (x, {}))

        # 展开元组数据（总分在前，明细在后）
        zt_df[['total_score', 'score_details']] = pd.DataFrame(zt_df['total_score'].tolist(), index=zt_df.index)

        # 过滤无效得分
        zt_df = zt_df[zt_df['total_score'].notna()]

        return zt_df.sort_values('total_score', ascending=False).head(self.params['max_candidates'])

    def _add_sector_info(self, zt_df, sector_map):
        """添加板块信息"""

        def match_sectors(code):
            return [sector for sector, codes in sector_map.items() if code in codes]

        zt_df['hot_sectors'] = zt_df['代码'].apply(match_sectors)
        return zt_df

    def _calculate_lhb_scores(self, lhb_data):
        """计算龙虎榜评分"""
        # 示例评分规则：每个机构席位加10分，每个游资席位加5分
        scores = {}
        for _, row in lhb_data.iterrows():
            code = row['代码']
            score = 0
            if '机构专用' in row['营业部名称']:
                score += row['营业部名称'].count('机构专用') * 10
            if any(b in row['营业部名称'] for b in ['华鑫', '中信']):
                score += row['营业部名称'].count(';') * 5
            scores[code] = score
        return scores

    def _calculate_total_score(self, row):
        """专为首板/二板设计的评分模型"""
        score_details = {}

        # 基础分 (50%)
        try:
            time_score = self._time_score(row['首次封板时间']) * 0.5  # 时间权重提升
            score_details['封板时间'] = time_score
        except:
            score_details['封板时间'] = 0

        # 量能分 (30%)
        try:
            # 量比得分
            vol_ratio = row.get('量比', 1)
            vol_score = min(vol_ratio * 15, 30)  # 量比>2得满分
            # 封单金额得分
            order_score = min(row['封板资金'] / 1e8 * 20, 20)  # 每亿加20分
            score_details['量能分'] = vol_score + order_score
        except:
            score_details['量能分'] = 0

        # 板块分 (15%)
        try:
            sector_score = min(len(row['hot_sectors']) * 10, 15)  # 最多加15分
            score_details['板块热度'] = sector_score
        except:
            score_details['板块热度'] = 0

        # 龙虎榜分 (5%)
        try:
            score_details['机构参与'] = min(row.get('lhb_score', 0) * 0.1, 5)  # 每机构席位加0.5分
        except:
            score_details['机构参与'] = 0

        # 特殊扣分项
        penalties = 0
        if row['炸板次数'] > 0:
            penalties -= row['炸板次数'] * 10  # 每次炸板扣10分

        total = sum(score_details.values()) + penalties
        return total, score_details

    def _time_score(self, first_time):
        """时间评分"""
        hour = int(first_time[:2])
        if hour < 9:
            return 100
        elif hour == 9:
            minute = int(first_time[3:5])
            return 100 - minute * 2
        else:
            return max(60 - (hour - 10) * 10, 0)

    def _create_trade_plan(self, trade_date, candidates, market_score):
        """生成带明细的交易计划"""
        plan = {
            'date': trade_date,
            'market_score': market_score,
            'candidates': []
        }

        for _, row in candidates.iterrows():
            total_score, score_details = self._calculate_total_score(row)
            candidate = {
                'code': row['代码'],
                'name': row['名称'],
                'price': round(row['最新价'] * 1.03, 2),
                'position': self._calculate_position(row, market_score),
                'total_score': total_score,
                'score_details': score_details,
                'reason': self._generate_reason(row)
            }
            plan['candidates'].append(candidate)

        plan['candidates'] = sorted(
            plan['candidates'],
            key=lambda x: x['total_score'],
            reverse=True
        )[:self.params['max_candidates']]
        return plan

    def _calculate_position(self, row, market_score):
        """首板/二板专用仓位策略"""
        base = 15  # 基础仓位

        # 封板时间加成 (早盘板加更多)
        time_bonus = {
            '09:25': 10, '09:30': 8, '09:45': 5,
            '10:00': 3, '14:30': 1
        }
        for t, score in time_bonus.items():
            if row['首次封板时间'] <= t:
                base += score
                break

        # 量能加成
        base += min(row.get('量比', 1) * 3, 9)  # 量比每1加3%，最高9%

        # 板块加成
        if row['hot_sectors']:
            base += min(len(row['hot_sectors']) * 2, 6)  # 每个关联板块加2%

        # 市场情绪调整
        if market_score > 70:
            base *= 1.2
        elif market_score < 50:
            base *= 0.8

        return min(base, 30)
    # def _calculate_position(self, row, market_score):
    #     """动态仓位计算"""
    #     base = self.params['position_limits']['default']
    #     # 市场情绪加成
    #     base += (market_score - 60) / 40 * 10  # 在60-100分区间线性加成0-10%
    #
    #     # 板块热度加成
    #     if len(row['hot_sectors']) > 0:
    #         base += min(len(row['hot_sectors']) * 5, 15)
    #
    #     # 连板次数加成
    #     base += min(row['limit_count'], 5) * 3  # 每个连板加3%
    #
    #     return min(base, self.params['position_limits']['sector_leader'])

    # def _generate_reason(self, row):
    #     """生成买入理由"""
    #     reasons = []
    #     # 连板信息
    #     if row['limit_count'] >= 1:
    #         reasons.append(f"{row['limit_count']}连板")
    #
    #     # 涨停时间
    #     if pd.notna(row['首次封板时间']):
    #         reasons.append(f"首次涨停时间：{row['首次封板时间']}")
    #
    #     # 炸板次数
    #     if row['炸板次数'] > 0:
    #         reasons.append(f"炸板{row['炸板次数']}次")
    #
    #     # 量能信息
    #     if '量比' in row and row['量比'] > 2:
    #         reasons.append(f"量比{row['量比']:.1f}倍")
    #
    #     # 龙虎榜明细
    #     lhb_details = []
    #     if row.get('lhb_score', 0) > 0:
    #         # 获取该股票的龙虎榜数据
    #         lhb_data = self._get_stock_lhb_data(row['代码'])
    #         if not lhb_data.empty:
    #             # 统计知名游资
    #             famous_count = lhb_data['营业部名称'].apply(
    #                 lambda x: any(name in x for name in ['华鑫', '中信', '东方财富'])
    #             ).sum()
    #             if famous_count > 0:
    #                 lhb_details.append(f"{famous_count}家知名席位")
    #
    #             # 买卖金额
    #             buy_amount = lhb_data['买入金额'].sum() / 1e8
    #             if buy_amount > 0:
    #                 lhb_details.append(f"净买入{buy_amount:.2f}亿")
    #
    #     if lhb_details:
    #         reasons.append("龙虎榜：" + "，".join(lhb_details))
    #
    #     # 板块信息
    #     if len(row['hot_sectors']) > 0:
    #         sector_info = []
    #         for sector, s_type in row['hot_sectors'][:2]:  # 最多展示两个板块
    #             momentum = self.sector_analyzer.get_sector_momentum(sector, s_type)
    #             sector_info.append(f"{sector}({momentum}分)")
    #         reasons.append("🔥热门板块: " + " | ".join(sector_info))
    #
    #     return " | ".join(reasons)

    def _generate_reason(self, row):
        """首板/二板专用买入理由"""
        reasons = []

        # 连板特征
        reasons.append(f"{row['limit_count']}连板")

        # 封板质量
        if row['首次封板时间'] < '10:00':
            reasons.append("早盘快速封板")
        if row['封板资金'] > 5e8:
            reasons.append("大单封死")

        # 量能特征
        if '量比' in row and row['量比'] > 2:
            reasons.append(f"量能充沛(量比{row['量比']:.1f})")

        # 资金动向
        lhb_info = []
        if row.get('lhb_score', 0) > 0:
            lhb_info.append("机构参与" if '机构专用' in str(row.get('营业部名称')) else "游资介入")
        if lhb_info:
            reasons.append("💰" + "+".join(lhb_info))

        # 板块驱动
        if row['hot_sectors']:
            top_sector = max(
                row['hot_sectors'],
                key=lambda x: self.sector_analyzer.get_sector_momentum(*x)
            )
            reasons.append(f"🔥{top_sector[0]}")

        return " | ".join(reasons)

    def _get_stock_lhb_data(self, code):
        """获取单只股票的龙虎榜明细"""
        if not hasattr(self, '_lhb_cache'):
            self._lhb_cache = self.lhb_processor.get_enhanced_data(
                dates=[self.data_date],
                statistic_period=self.period
            )
        return self._lhb_cache[self._lhb_cache['代码'] == code]

    def _generate_empty_plan(self, date, message):
        """生成空计划"""
        return {
            'date': date,
            'market_score': 0,
            'candidates': [],
            'message': message
        }


if __name__ == "__main__":
    system = TradingSystem()
    plan = system.run()
    candidates = plan.get('candidates', [])

    # 查看首板股票
    first_limit = [c for c in candidates if c.get('连板数') == 1]
    print(f"优选首板股：{len(first_limit)}只")
    for stock in first_limit:
        print(f"""
        ► {stock['名称']} [{stock['代码']}]
        封板时间：{stock['首次封板时间']}
        封单金额：{stock['封板资金'] / 1e8:.2f}亿
        入选理由：{stock['reason']}
        """)
    # analyzer = SectorAnalyzer(trade_date=plan['date'])
    #
    # # 获取近2日热门板块前5
    # hot_sectors = analyzer.get_hot_sectors(days=2, top_n_per_type=5)
    #
    # print(f"{plan['date']} 交易计划")
    # # 打印报告时的优化显示
    #
    # Mkt_analyzer = MarketSentimentAnalyzer()
    # report = Mkt_analyzer.generate_report()
    # print(f"【市场情绪日报】{report['交易日期']}")
    # print(f"综合评分: {report['综合情绪分']} ({report['情绪级别']})")
    # print("核心维度:")
    # for k, v in report['得分明细'].items():
    #     print(f"- {k}: {v}")
    #
    # print(f"最高连板数: {report['市场数据']['涨停分析']['最高连板数']}")
    # print("连板分布:")
    # for k, v in report['市场数据']['涨停分析']['连板分布'].items():
    #     print(f"  {k}: {v}家")
    #
    # if report['市场数据']['涨停分析']['特殊涨停案例']:
    #     print("\n📌 非连续涨停案例:")
    #     for case in report['市场数据']['涨停分析']['特殊涨停案例']:
    #         print(f"  - {case}")
    # # 检测极端案例
    # extreme_data = Mkt_analyzer.detect_extreme_boards()
    # print(f"检测到天地板：{extreme_data['sky_earth']}例，地天板：{extreme_data['earth_sky']}例,{extreme_data['details']}")
    #
    # print("热门板块:", hot_sectors)
    # if plan.get('candidates'):
    #     for stock in plan['candidates']:
    #         print(f"► {stock['name']}({stock['code']})")
    #         print(f"  报价：{stock['price']} | 仓位：{stock['position']}%")
    #         print(f"  总分：{stock['total_score']}")
    #
    #         # 得分明细展示
    #         if system.show_score_details:
    #             details = [
    #                 f"{k}: {v:+}"
    #                 for k, v in stock['score_details'].items()
    #             ]
    #             print("  得分明细：", " | ".join(details))
    #
    #         print(f"  理由：{stock['reason']}\n")
    # else:
    #     print(f"⚠️ 今日无交易计划：{plan.get('message', '')}")