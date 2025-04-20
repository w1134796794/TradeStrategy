import pandas as pd
import logging
from datetime import datetime
import os
from GetHotSectors import SectorAnalyzer
from GetMarketSentiment import MarketSentimentAnalyzer
from GetTradeDate import LocalTradeCalendar


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradePlanGenerator:
    """交易计划生成器"""
    def __init__(self, data_root: str):
        self.data_dir = data_root
        self.market_sentiment = None
        self.hot_sectors = None
        self.dragon_tiger_data = None
        self.zt_data = None
        self.sector_components = None
        self.SectorAnalyzer = SectorAnalyzer(data_root)
        self.MarketAnalyzer = MarketSentimentAnalyzer(data_root=data_root)
        self.calendar = LocalTradeCalendar()

    def load_data(self):
        """从CSV文件加载数据"""
        # 加载市场情绪数据
        self.market_sentiment = self.MarketAnalyzer.generate_report()

        # 加载热门板块数据
        self.hot_sectors = self.SectorAnalyzer.generate_hot_sectors()

        # 加载龙虎榜数据
        dragon_tiger_path = os.path.join(self.data_dir, 'lhb_main.csv')
        if os.path.exists(dragon_tiger_path):
            self.dragon_tiger_data = pd.read_csv(dragon_tiger_path)
            logger.info(f"加载龙虎榜数据: {dragon_tiger_path}")
        else:
            logger.error(f"龙虎榜数据文件不存在: {dragon_tiger_path}")
            return False

        # 加载龙虎榜席位
        dragon_seat_path = os.path.join(self.data_dir, 'lhb_detail.csv')
        if os.path.exists(dragon_seat_path):
            self.dragon_seat = pd.read_csv(dragon_seat_path)

            self.dragon_seat["数据日期"] = self.dragon_seat["数据日期"].astype(str)
            self.dragon_seat["股票代码"] = self.dragon_seat["股票代码"].astype(str).str.zfill(6)
            logger.info(f"加载龙虎榜数据: {dragon_seat_path}")
        else:
            logger.error(f"龙虎榜明细数据文件不存在: {dragon_seat_path}")
            return False

        # 加载涨停股票数据
        zt_path = os.path.join(self.data_dir, 'zt_pool_hist.csv')
        if os.path.exists(zt_path):
            self.zt_data = pd.read_csv(zt_path)

            self.zt_data["日期"] = self.zt_data["日期"].astype(str)
            self.zt_data["代码"] = self.zt_data["代码"].astype(str).str.zfill(6)

            logger.info(f"加载涨停股票数据: {zt_path}")
        else:
            logger.error(f"涨停股票数据文件不存在: {zt_path}")
            return False

        # 加载板块成分股数据
        sector_components_path = os.path.join(self.data_dir, 'sector_components.csv')
        if os.path.exists(sector_components_path):
            self.sector_components = pd.read_csv(sector_components_path)
            self.sector_components["板块代码"] = self.sector_components["板块代码"].astype(str)
            self.sector_components["代码"] = self.sector_components["代码"].astype(str).str.zfill(6)
            logger.info(f"加载板块成分股数据: {sector_components_path}")
        else:
            logger.error(f"板块成分股数据文件不存在: {sector_components_path}")
            return False

        return True

    def get_hot_dragons(self):
        """获取所有热门板块的龙头股（新增方法）"""
        dragons = []
        for sector in self.hot_sectors:
            sector_name, sector_type = sector[0], sector[1]
            dragons += self.SectorAnalyzer.generate_sector_dragons(sector_name, sector_type)
        return {d['code']: d['total_score'] for d in dragons}

    def analyze_dragon_seat(self):
        """龙虎榜席位分析（新增方法）"""
        try:
            # 预处理数据
            self.dragon_seat['数据日期'] = pd.to_datetime(self.dragon_seat['数据日期'], format='%Y%m%d')
            self.dragon_seat['next_3d_pct'] = 0.0

            # 计算席位胜率
            seat_stats = []
            for seat_name, group in self.dragon_seat.groupby('交易营业部名称'):
                win_count = 0
                total = len(group)
                for _, row in group.iterrows():
                    # 获取上榜后3日行情
                    start_date = row['数据日期'] + pd.Timedelta(days=1)
                    end_date = row['数据日期'] + pd.Timedelta(days=3)
                    hist = self.SectorAnalyzer.fetch_sector_stock_hist(
                        row['股票代码'],
                        start_date.strftime('%Y%m%d'),
                        end_date.strftime('%Y%m%d')
                    )
                    if not hist.empty:
                        pct_change = (hist['收盘'].iloc[-1] / hist['开盘'].iloc[0] - 1) * 100
                        win_count += 1 if pct_change > 0 else 0
                win_rate = win_count / total if total > 0 else 0
                seat_stats.append({'seat': seat_name, 'win_rate': win_rate, 'count': total})

            # 筛选有效数据
            df_seat = pd.DataFrame(seat_stats)
            df_seat = df_seat[df_seat['count'] >= 5]  # 至少出现5次

            # 排序并保存结果
            self.top_seats = df_seat.nlargest(5, 'win_rate')['seat'].tolist()
            self.bottom_seats = df_seat.nsmallest(5, 'win_rate')['seat'].tolist()

            logger.info(f"顶级席位：{self.top_seats}")
            logger.info(f"低胜率席位：{self.bottom_seats}")

        except Exception as e:
            logger.error(f"龙虎榜分析失败：{str(e)}")
            self.top_seats = []
            self.bottom_seats = []

    def generate_trade_plan(self):
        """生成交易计划"""
        if not self.load_data():
            logger.error("数据加载失败，无法生成交易计划")
            return None

        # 新增龙虎榜分析
        self.analyze_dragon_seat()

        # 获取热门板块龙头股
        self.hot_dragons = self.get_hot_dragons()

        # 筛选候选股
        candidates = self.screen_candidates()

        if candidates.empty:
            logger.warning("没有找到符合条件的候选股")
            return None

        # 评分系统
        scored_candidates = self.score_candidates(candidates)

        # 排序并选择前5名
        top_candidates = self.select_top_candidates(scored_candidates)

        # 生成交易计划
        trade_plan = self.compile_trade_plan(top_candidates)

        return trade_plan

    def screen_candidates(self):
        """筛选候选股"""
        # 筛选当天涨停的股票
        today = self.calendar.get_recent_trade_date()
        candidates = self.zt_data[self.zt_data['日期'] == today]

        if candidates.empty:
            logger.warning("没有找到当天涨停的股票")
            return pd.DataFrame()

        return candidates

    def score_candidates(self, candidates):
        """对候选股进行评分"""
        scored_candidates = []
        for _, candidate in candidates.iterrows():
            score = 0
            reasons = []

            # 1. 大盘情绪
            market_score = self.market_sentiment['total_score']
            if market_score > 20:
                score += 20  # 大盘情绪良好加分
                reasons.append(f"大盘情绪良好（评分：{market_score}）")

            # 2. 热门板块
            # 筛选热门板块的成分股
            hot_sector_stocks = self.get_hot_sector_stocks()
            if candidate['代码'] in hot_sector_stocks:
                score += 15  # 热门板块加分
                reasons.append("属于热门板块")

            # 2.1 龙头股额外加分
            dragon_score = self.hot_dragons.get(candidate['代码'], 0)
            if dragon_score > 0:
                score += dragon_score * 0.2  # 按龙头股原始得分的20%加成
                reasons.append(f"板块龙头股（强度：{dragon_score:.1f}）")

            # 2.2 龙虎榜席位分析
            seat_info = self.dragon_seat[self.dragon_seat['股票代码'] == candidate['代码']]
            if not seat_info.empty:
                # 计算席位得分
                seat_scores = []
                for seat in seat_info['交易营业部名称']:
                    if seat in self.top_seats:
                        seat_scores.append(3)  # 顶级席位+3分
                    elif seat in self.bottom_seats:
                        seat_scores.append(-2) # 低胜率席位-2分
                if seat_scores:
                    seat_total = sum(seat_scores)
                    score += seat_total
                    reasons.append(f"龙虎榜席位得分：{seat_total}")

            # 4. 涨停时间
            if pd.notna(candidate['首次封板时间']):
                # 确保时间是字符串格式
                seal_time_str = str(int(candidate['首次封板时间'])).zfill(6)
                try:
                    seal_time = datetime.strptime(seal_time_str, '%H%M%S').time()
                    if seal_time.hour < 10:
                        score += 15  # 上午快速封板加分
                        reasons.append("上午快速封板")
                    elif seal_time.hour < 13:
                        score += 10  # 下午早期封板加分
                        reasons.append("下午早期封板")
                    else:
                        score += 5  # 下午晚期封板加分
                        reasons.append("下午晚期封板")
                except ValueError:
                    logger.error(f"时间格式错误: {seal_time_str}")

            # 5. 其他因素（如换手率、流通市值等）
            if pd.notna(candidate['流通市值']):
                if candidate['流通市值'] < 30e8:
                    score += 15  # 小市值加分
                    reasons.append("小市值股票，流动性较好")

            scored_candidates.append({
                'stock_code': candidate['代码'],
                'stock_name': candidate['名称'],
                'score': score,
                'reasons': reasons
            })

        return scored_candidates

    def get_hot_sector_stocks(self):
        """获取热门板块的成分股"""
        hot_sector_stocks = set()
        for sector in self.hot_sectors:
            sector_name = sector[0]
            sector_type = sector[1]
            # 筛选热门板块的成分股
            sector_stocks = self.sector_components[
                (self.sector_components['板块名称'] == sector_name) &
                (self.sector_components['板块类型'] == sector_type)
            ]['代码'].tolist()
            hot_sector_stocks.update(sector_stocks)
        return hot_sector_stocks

    def select_top_candidates(self, scored_candidates):
        """选择评分最高的前5名股票"""
        if not scored_candidates:
            return []

        # 按评分排序
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        # 选择前5名
        return scored_candidates[:5]

    def compile_trade_plan(self, top_candidates):
        """编制交易计划"""
        if not top_candidates:
            return None

        trade_plan = {
            'date': datetime.now().strftime('%Y%m%d'),
            'candidates': top_candidates
        }
        return trade_plan


# 使用示例
if __name__ == "__main__":
    data_root = 'data/csv/20250418'
    generator = TradePlanGenerator(data_root)
    trade_plan = generator.generate_trade_plan()
    if trade_plan:
        print(f"交易计划日期: {trade_plan['date']}")
        print("推荐股票:")
        for candidate in trade_plan['candidates']:
            print(f"股票代码: {candidate['stock_code']}, 股票名称: {candidate['stock_name']}, 评分: {candidate['score']}")
            print(f"理由: {', '.join(candidate['reasons'])}")
            print()
    else:
        print("未能生成交易计划")
