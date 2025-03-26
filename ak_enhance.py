from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from venv import logger

import akshare as ak
from datetime import datetime, timedelta


class PreOpenLimitUpStrategy:
    def __init__(self):
        self.hot_sectors = []
        self.trade_dates = self._load_trade_dates()

        # 评分参数
        self.scoring_params = {
            'time_weights': {
                '09:30:00': 100,
                '09:40:00': 90,
                '10:00:00': 80,
                '10:30:00': 60,
                '13:00:00': 40,
                '14:00:00': 20
            },
            'order_amount_weight': 20,  # 每1亿加20分
            'reopen_penalty': 30,  # 每次炸板扣30分
            'sector_count_bonus': 15,  # 每个匹配板块加15分
            'main_sector_bonus': 30,  # 前3热门板块额外加30分
            'leader_bonus': 50  # 板块龙头额外加50分
        }

    def get_zt_pool_data(self):
        """获取涨停板核心数据"""
        zt_data = ak.stock_zt_pool_em(date=self.trade_date)
        # 字段重命名
        column_map = {
            '代码': 'code',
            '名称': 'name',
            '最新价': 'close',
            '首次封板时间': 'first_time',
            '最后封板时间': 'last_time',
            '炸板次数': 'reopen_count',
            '涨停统计': 'limit_count',
            '封单资金': 'order_amount',
            '流通市值': 'float_mv',
            '换手率': 'turnover_rate'
        }
        return zt_data.rename(columns=column_map)

    def calculate_strength_score(self, row):
        """计算个股涨停强度评分"""
        score = 0

        # 封板时间评分（越早越好）
        time_score = {
            '09:30:00': 100,
            '09:40:00': 90,
            '10:00:00': 80,
            '10:30:00': 60,
            '13:00:00': 40,
            '14:00:00': 20
        }
        for t, s in time_score.items():
            if row['first_time'] <= t:
                score += s
                break

        # 封单资金评分（单位：亿）
        score += min(row['order_amount'] / 1e8 * 20, 100)  # 每1亿加20分

        # 炸板次数惩罚
        score -= row['reopen_count'] * 30

        # 流通市值评分（20-50亿最优）
        if 20e8 <= row['float_mv'] <= 50e8:
            score += 50
        elif 50e8 < row['float_mv'] <= 100e8:
            score += 30
        else:
            score += 10

        # 换手率修正
        if row['turnover_rate'] < 5:
            score += 20
        elif 5 <= row['turnover_rate'] < 15:
            score += 10
        else:
            score -= 10

        return score

    def get_hot_sectors(self,days=3):
        """获取三日强度最高的板块"""
        # 行业板块分析
        industry = ak.stock_board_industry_hist_em()
        industry['3日强度'] = industry['最新价'].pct_change(3)
        hot_industries = industry.sort_values('3日强度',ascending=False).head(5)['板块名称'].tolist()

        # 概念分析
        concept = ak.stock_board_concept_hist_em()
        concept['3日强度'] = concept['最新价'].pct_change(3)
        hot_concepts = concept.sort_values('3日强度', ascending=False).head(5)['板块名称'].tolist()

        return list(set(hot_industries + hot_concepts))

    def enhance_sector_analysis(self):
        """增强版板块分析模块"""
        # 获取当日涨停股票
        zt_df = self.get_zt_pool_data()

        # 获取热门板块（3日强度前5）
        hot_sectors = self.get_hot_sectors(days=3)

        # 创建板块-成分股映射（带缓存）
        sector_stocks_map = self._create_sector_map(hot_sectors)

        # 为每个涨停股打标签
        zt_df['hot_sector'] = zt_df['code'].apply(
            lambda x: self._check_sector_membership(x, sector_stocks_map))

        return zt_df

    def _create_sector_map(self, hot_sectors):
        """创建板块成分股映射表（优化性能）"""
        sector_map = {}

        # 并行获取板块成分股
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(ak.stock_board_cons_em, sector): sector
                       for sector in hot_sectors}
            for future in as_completed(futures):
                sector = futures[future]
                try:
                    data = future.result()
                    sector_map[sector] = set(data['代码'].tolist())
                except Exception as e:
                    logger.error(f"板块{sector}成分股获取失败: {str(e)}")

        return sector_map

    def _check_sector_membership(self, code, sector_map):
        """判断股票所属热门板块"""
        matched_sectors = []

        # 获取该股票所属全部板块
        stock_sectors = ak.stock_board_concept_name_ths()['概念名称'].tolist()

        # 匹配热门板块
        for sector, stocks in sector_map.items():
            if code in stocks or sector in stock_sectors:
                matched_sectors.append(sector)

        return matched_sectors

    def calculate_base_score(self, row):
        """基础评分逻辑"""
        score = 0

        # 封板时间评分
        score += self._calc_time_score(row['first_time'])

        # 封单金额评分
        score += min(row['order_amount'] / 1e8 * self.scoring_params['order_amount_weight'], 100)

        # 炸板次数惩罚
        score -= row['reopen_count'] * self.scoring_params['reopen_penalty']

        # 流通市值评分
        score += self._calc_market_cap_score(row['float_mv'])

        # 换手率修正
        score += self._calc_turnover_score(row['turnover_rate'])

        return score

    def calculate_enhanced_score(self, row):
        """增强评分（含板块因子）"""
        base_score = self.calculate_base_score(row)

        # 板块相关加分
        sector_bonus = len(row['hot_sector']) * self.scoring_params['sector_count_bonus']
        main_sector_bonus = self.scoring_params['main_sector_bonus'] if any(
            s in self.hot_sectors[:3] for s in row['hot_sector']) else 0

        # 龙头股识别
        leader_bonus = self.scoring_params['leader_bonus'] if self._is_sector_leader(row) else 0

        return base_score + sector_bonus + main_sector_bonus + leader_bonus

    def _calc_time_score(self, first_time):
        """计算封板时间得分"""
        for t, s in self.scoring_params['time_weights'].items():
            if first_time <= t:
                return s
        return 0

    def generate_trade_plan(self):
        """生成次日交易计划"""
        # 获取基础数据
        df = self.get_zt_pool_data()

        # 过滤无效数据
        df = df[df['close'] > 0]

        # 计算强度评分
        df['strength_score'] = df.apply(self.calculate_strength_score, axis=1)

        # 添加板块强度
        df['sector_strength'] = df['code'].apply(self.analyze_sector_strength)

        # 综合排序
        df['final_score'] = df['strength_score'] * 0.7 + df['sector_strength'] * 0.3
        df = df.sort_values('final_score', ascending=False)

        # 生成买入列表
        buy_list = []
        for _, row in df.head(20).iterrows():  # 取前20名进一步筛选
            # 排除高风险特征
            if row['reopen_count'] >= 2 or row['order_amount'] < 5e7:
                continue

            buy_list.append({
                'code': row['code'],
                'name': row['name'],
                'score': round(row['final_score'], 1),
                'price': round(row['close'] * 1.03, 2),  # 溢价3%报价
                'position': self.calculate_position(row),
                'reason': self.generate_reason(row)
            })

        return sorted(buy_list, key=lambda x: x['score'], reverse=True)[:5]  # 最终选取前5

    def calculate_position(self, row):
        """仓位计算模型"""
        base = 20  # 基础仓位20%
        # 封单加成
        base += min(row['order_amount'] / 1e8 * 5, 15)
        # 时间加成
        if row['first_time'] <= '09:40:00':
            base += 15
        elif row['first_time'] <= '10:00:00':
            base += 10
        # 板块加成
        base += min(row['sector_strength'] * 2, 10)
        return min(base, 40)  # 单票最大仓位40%

    def generate_reason(self, row):
        """生成买入理由"""
        reasons = []
        if row['first_time'] <= '09:40:00':
            reasons.append('早盘封板')
        if row['order_amount'] > 1e8:
            reasons.append(f"大单封板({row['order_amount'] / 1e8:.1f}亿)")
        if row['reopen_count'] == 0:
            reasons.append('零炸板')
        if row['sector_strength'] >= 5:
            reasons.append(f"板块强势({row['sector_strength']}家涨停)")
        return "，".join(reasons)

    def execute(self):
        """执行策略"""
        plan = self.generate_trade_plan()
        print("次日竞价买入计划：")
        for idx, item in enumerate(plan, 1):
            print(f"{idx}. {item['name']}({item['code']})")
            print(f"   报价：{item['price']}元")
            print(f"   建议仓位：{item['position']}%")
            print(f"   入选理由：{item['reason']}\n")

    def analyze_sector_structure(self, sector):
        """分析板块涨停梯队"""
        zt_stocks = self.get_zt_pool_data()
        sector_stocks = zt_stocks[zt_stocks['所属板块'] == sector]

        structure = {
            '首板数量': len(sector_stocks[sector_stocks['连板数'] == 1]),
            '连板高度': sector_stocks['连板数'].max()
        }

        # 加分逻辑：板块有3只以上首板或存在高度龙头
        if structure['首板数量'] >= 3:
            return 20
        elif structure['连板高度'] >= 3:
            return 15
        else:
            return 5

    def get_market_mood(self):
        """获取市场整体情绪"""
        zt_count = len(self.get_zt_pool_data())
        dt_count = len(ak.stock_zt_pool_dtgc_em(date=self.trade_date))

        return {
            '涨停封板率': zt_count / (zt_count + dt_count) if zt_count + dt_count > 0 else 0,
            '昨日涨停收益': self.calculate_zt_yield()
        }

    def calculate_zt_yield(self):
        """计算昨日涨停股今日平均收益率"""
        prev_date = (datetime.strptime(self.trade_date, "%Y%m%d") - timedelta(1)).strftime("%Y%m%d")
        prev_zt = ak.stock_zt_pool_em(date=prev_date)
        today_data = ak.stock_zh_a_spot_em()

        merged = prev_zt.merge(today_data[['代码', '涨跌幅']], on='代码')
        return merged['涨跌幅'].mean()


# 执行示例
if __name__ == "__main__":
    strategy = PreOpenLimitUpStrategy()
    strategy.execute()