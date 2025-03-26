from concurrent.futures import ThreadPoolExecutor
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import logging
from TradeDate import TradeCalendar

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PreOpenLimitUpStrategy:
    def __init__(self):
        # 获取最近的交易日（跳过非交易日）
        calendar = TradeCalendar()
        self.trade_date = calendar.get_recent_trade_date()
        self.hot_sectors = []
        self.sector_stocks_map = {}
        self.dragon_tiger_df = pd.DataFrame()  # 龙虎榜数据缓存

        # 更新评分参数
        self.scoring_params = {
            # ...保留原有参数...
            'dragon_tiger_weights': {
                '知名游资': 20,  # 每个知名游资席位
                '拉萨军团': -10,  # 每个拉萨营业部
                '机构专用': 15  # 机构席位
            }
        }

        # 重写市场情绪参数
        self.market_params = {
            'index_weights': {
                'sh000001': 0.3,  # 上证指数
                'sz399006': 0.2,  # 创业板指
                'sh000016': 0.2,  # 上证50
                'sh000905': 0.3  # 中证500
            },
            'sentiment_threshold': 60,
            '基础情绪分': 50,
            '跌停惩罚阈值': 30,  # 跌停数超过30个时启动惩罚
            'vol_compare_window': 5  # 成交量对比窗口
        }

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
            'leader_bonus': 50,  # 板块龙头额外加50分
            'limit_count_weights': {
                2: 20,  # 2连板
                3: 40,  # 3连板
                4: 60,  # 4连板
                5: 100  # 5连板及以上
            },
            'continuous_limit_bonus': 15,  # 每增加1板超过5板的额外加分

        }

    def get_zt_pool_data(self):
        """获取涨停板核心数据并进行数据清洗"""
        try:
            zt_data = ak.stock_zt_pool_em(date=self.trade_date)
            # 列名标准化处理
            column_map = {
                '代码': 'code',
                '名称': 'name',
                '最新价': 'close',
                '首次封板时间': 'first_time',
                '最后封板时间': 'last_time',
                '炸板次数': 'reopen_count',
                '涨停统计': 'limit_count',
                '封板资金': 'order_amount',
                '流通市值': 'float_mv',
                '换手率': 'turnover_rate'
            }
            # 处理可能的列名差异
            zt_data = zt_data.rename(columns={k: v for k, v in column_map.items() if k in zt_data.columns})

            # 类型转换与单位处理（假设akshare返回的单位是万元）
            zt_data['order_amount'] = zt_data['order_amount'] * 1e4  # 转换为元
            zt_data['float_mv'] = zt_data['float_mv'] * 1e8  # 转换为元

            # 新增：解析涨停统计字段（格式示例："8/5" 表示8天5板）
            if '涨停统计' in zt_data.columns:
                zt_data[['total_days', 'limit_count']] = zt_data['涨停统计'].str.split('/', expand=True).astype(int)
            elif 'limit_count' in zt_data.columns:  # 兼容不同数据源
                zt_data[['total_days', 'limit_count']] = zt_data['limit_count'].str.split('/', expand=True).astype(int)

            return zt_data
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return pd.DataFrame()

    def get_dragon_tiger_data(self, days=3):
        """优化版龙虎榜数据获取"""
        try:
            # 获取日期范围（最近N个交易日）

            calendar = TradeCalendar()
            end_date = self.trade_date
            start_date = calendar.get_previous_trade_date(end_date, days - 1)

            # 验证日期有效性
            if not (start_date and end_date and start_date <= end_date):
                logger.error(f"无效日期范围: {start_date} - {end_date}")
                return pd.DataFrame()

            # 单次请求获取多日数据
            df = ak.stock_lhb_detail_em(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )

            # 数据清洗
            if not df.empty:
                # 统一日期格式处理
                df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime("%Y%m%d")
                # 字段重命名
                column_map = {
                    '代码': 'code',
                    '名称': 'name',
                    '买入营业部名称': 'buy_branches',
                    '卖出营业部名称': 'sell_branches',
                    'trade_date': 'trade_date'
                }
                return df.rename(columns=column_map)[list(column_map.values())]
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"龙虎榜数据获取失败: {str(e)}", exc_info=True)
            return pd.DataFrame()

    def _calc_dragon_tiger_score(self, code):
        """龙虎榜评分逻辑"""
        if self.dragon_tiger_df.empty:
            self.dragon_tiger_df = self.get_dragon_tiger_data()

        # 获取该股票最近3天的龙虎榜记录
        records = self.dragon_tiger_df[self.dragon_tiger_df['code'] == code]
        if records.empty:
            return 0

        score = 0
        # 定义营业部规则（示例名单可根据需要扩展）
        branch_rules = {
            '知名游资': ['华鑫证券上海分公司', '中信证券上海溧阳路', '国泰君安南京太平南路'],
            '拉萨军团': ['东方财富拉萨团结路', '东方财富拉萨东环路'],
            '机构专用': ['机构专用']
        }

        for _, record in records.iterrows():
            # 分析买入营业部
            branches = [b.strip() for b in record['buy_branches'].split(';') if b.strip()]
            for branch in branches:
                for category in branch_rules:
                    if any(key in branch for key in branch_rules[category]):
                        score += self.scoring_params['dragon_tiger_weights'][category]
                        break  # 每个营业部只匹配一个类别
        return score

    def get_hot_sectors(self, days=2):
        """获取近期强势板块（行业+概念）"""
        trade_date_obj = datetime.strptime(self.trade_date, "%Y%m%d")  # 字符串转日期对象
        start_date = (trade_date_obj - timedelta(days=days)).strftime("%Y%m%d")
        end_date = self.trade_date
        try:
            # 获取行业板块排行（当日涨幅降序）
            industry_list = ak.stock_board_industry_name_em()["板块名称"].tolist()
            industry_rank = []
            for industry in industry_list:
                temp_df = ak.stock_board_industry_hist_em(symbol=industry, start_date=start_date, end_date=end_date,
                                                          period="日k", adjust="")
                industry_rank.append({
                    "name": industry,
                    "type": "industry",
                    "score": temp_df["涨跌幅"].iloc[-2:].sum()  # 2日累计涨幅
                })

            # 获取概念板块排行
            concept_list = ak.stock_board_concept_name_em()["板块名称"].tolist()
            concept_rank = []
            for concept in concept_list:
                temp_df = ak.stock_board_concept_hist_em(symbol=concept, period="daily", start_date=start_date, end_date=end_date, adjust="")
                concept_rank.append({
                    "name": concept,
                    "type": "concept",
                    "score": temp_df["涨跌幅"].iloc[-2:].sum()
                })

            all_sectors = pd.DataFrame(industry_rank + concept_rank)
            top_sectors = all_sectors.nlargest(10, 'score')

            # 合并去重并取前5
            # hot_sectors = list(set(top_industry + top_concept))

            #
            return [(row['name'], row['type']) for _, row in top_sectors.iterrows()]

        except Exception as e:
            logger.error(f"获取热门板块失败: {str(e)}", exc_info=True)
            return []

    def _create_sector_map(self, hot_sectors):
        """并行获取板块成分股数据"""
        sector_map = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            for sector_name, sector_type in hot_sectors:
                future = executor.submit(self._get_sector_stocks, sector_name, sector_type)
                futures[future] = (sector_name, sector_type)

            for future in futures:
                sector_name, sector_type = futures[future]
                try:
                    codes = future.result()
                    sector_map[(sector_name, sector_type)] = set(codes)
                except Exception as e:
                    logger.warning(f"板块{sector_name}({sector_type})成分股获取失败: {str(e)}")
        return sector_map

    def _get_sector_stocks(self, sector_name, sector_type):
        """获取板块成分股（带重试机制）"""
        try:
            if sector_type == "industry":
                data = ak.stock_board_industry_cons_em(symbol=sector_name)
            elif sector_type == "concept":
                data = ak.stock_board_concept_cons_em(symbol=sector_name)
            else:
                return []
            return data['代码'].tolist()
        except Exception as e:
            logger.error(f"获取{sector_type}板块{sector_name}成分股失败: {str(e)}")
            return []

    def enhance_sector_analysis(self):
        """执行板块分析并返回增强数据"""
        self.hot_sectors = self.get_hot_sectors()  # 现在返回的是带类型的列表
        if not self.hot_sectors:
            return pd.DataFrame()

        self.sector_stocks_map = self._create_sector_map(self.hot_sectors)
        zt_df = self.get_zt_pool_data()

        if zt_df.empty:
            return zt_df

        # 添加板块信息（同时匹配行业和概念）
        def find_related_sectors(code):
            related = []
            for (sector_name, sector_type), codes in self.sector_stocks_map.items():
                if code in codes:
                    related.append(f"{sector_name}({sector_type})")
            return related

        zt_df['hot_sectors'] = zt_df['code'].apply(find_related_sectors)
        return zt_df

    def calculate_enhanced_score(self, row):
        """综合评分模型"""
        # if not row or pd.isnull(row.get('order_amount')):
        #     return 0
        if pd.isnull(row.get('order_amount')):
            return 0

        score = 0
        # 基础评分
        score += self._calc_time_score(row.get('first_time', ''))
        score += min(row['order_amount'] / 1e8 * self.scoring_params['order_amount_weight'], 100)
        score -= row.get('reopen_count', 0) * self.scoring_params['reopen_penalty']
        score += self._calc_market_cap_score(row.get('float_mv', 0))
        score += self._calc_turnover_rate_score(row.get('turnover_rate', 0))

        # 新增连板次数评分
        score += self._calc_limit_count_score(row.get('limit_count', 0))

        # 板块加分
        sectors = row.get('hot_sectors', [])
        score += len(sectors) * self.scoring_params['sector_count_bonus']
        if any(s in self.hot_sectors[:3] for s in sectors):
            score += self.scoring_params['main_sector_bonus']
        if self._is_sector_leader(row):
            score += self.scoring_params['leader_bonus']
        # 加入龙虎榜评分
        score += self._calc_dragon_tiger_score(row['code'])

        return score

    def _calc_limit_count_score(self, limit_count):
        """计算连板次数得分"""
        try:
            # 确保输入为整数
            if isinstance(limit_count, str):
                limit_count = int(limit_count.split('/')[-1])  # 取后半部分数值
            limit_count = int(limit_count)
        except (ValueError, TypeError, AttributeError):
            return 0

            # 排除异常值
        limit_count = max(min(limit_count, 10), 0)  # 限制在0-10之间

        weights = self.scoring_params['limit_count_weights']
        base_score = 0

        # 阶梯式评分
        for threshold in sorted(weights.keys(), reverse=True):
            if limit_count >= threshold:
                base_score = weights[threshold]
                break

        # 超额奖励（5板后每多1板加15分）
        if limit_count > 5:
            extra = (limit_count - 5) * self.scoring_params['continuous_limit_bonus']
            return base_score + extra
        return base_score


    def _calc_time_score(self, first_time):
        """计算时间得分"""
        for t, s in self.scoring_params['time_weights'].items():
            if first_time <= t:
                return s
        return 0

    def _calc_market_cap_score(self, float_mv):
        """流通市值评分（单位：元）"""
        mv = float_mv / 1e8  # 转换为亿元
        if 20 <= mv <= 50:
            return 50
        elif 50 < mv <= 100:
            return 30
        else:
            return 10

    def _calc_turnover_rate_score(self, turnover_rate):
        """换手率评分"""
        if turnover_rate < 5:
            return 20
        elif 5 <= turnover_rate < 15:
            return 10
        else:
            return -10

    def _is_sector_leader(self, row):
        """判断是否板块龙头（简化版）"""
        if not row['hot_sectors']:
            return False
        # 获取同板块股票数据
        sector = row['hot_sectors'][0]
        sector_stocks = self.sector_stocks_map.get(sector, [])
        if not sector_stocks:
            return False
        # 检查是否连板高度最高
        zt_data = self.get_zt_pool_data()
        sector_data = zt_data[zt_data['code'].isin(sector_stocks)]
        max_limit = sector_data['limit_count'].max()
        return row['limit_count'] == max_limit

    def get_market_sentiment(self):
        """重写市场情绪数据获取"""
        # 获取基础市场数据
        try:
            # 获取指数数据
            # 安全获取市场广度数据
            market_breadth = {}
            breadth_items = {
                'rise_num': '上涨',
                'fall_num': '下跌',
                'limit_up': '真实涨停',
                'limit_down': '真实跌停'
            }

            index_data = {}
            for index_code in self.market_params['index_weights']:
                df = ak.stock_zh_index_daily(symbol=index_code)
                if df.empty:
                    continue
                df = df.iloc[-30:]  # 取最近30个交易日数据
                df['ma5'] = df['close'].rolling(5).mean()
                df['ma20'] = df['close'].rolling(20).mean()
                last = df.iloc[-1]
                index_data[index_code] = {
                    'change_pct': (last['close'] - last['open']) / last['open'] * 100,
                    'position': 'above' if last['close'] > last['ma5'] else 'below',
                    'trend': 'up' if last['ma5'] > last['ma20'] else 'down'
                }

            mb_df = ak.stock_market_activity_legu()
            for key, item_name in breadth_items.items():
                try:
                    value = mb_df[mb_df['item'] == item_name]['value'].iloc[0]
                    market_breadth[key] = int(value)
                except (IndexError, KeyError, ValueError):
                    logger.warning(f"市场广度字段 {item_name} 获取失败，使用默认值0")
                    market_breadth[key] = 0

            # 获取连板股数据
            zt_data = self.get_zt_pool_data()
            limit_counts = zt_data['limit_count'].value_counts().to_dict()
            max_limit = zt_data['limit_count'].max() if not zt_data.empty else 0

            return {
                'index_data': index_data,
                'market_breadth': market_breadth,
                'limit_stats': {
                    'max_limit': max_limit,
                    'distribution': limit_counts
                },
                'trade_date': self.trade_date
            }
        except Exception as e:
            logger.error(f"市场情绪数据获取失败: {str(e)}")
            return None

    def calculate_market_score(self, market_data):
        """重写市场情绪评分模型"""
        if not market_data:
            return 0

        score = self.market_params['基础情绪分']

        # 指数维度（权重40%）
        index_score = 0
        for code, data in market_data['index_data'].items():
            weight = self.market_params['index_weights'].get(code, 0)
            # 趋势评分
            trend_score = 20 if data['trend'] == 'up' else 0
            # 位置评分
            position_score = 10 if data['position'] == 'above' else 0
            # 涨跌幅评分
            change_score = max(min(data['change_pct'] * 2, 20), -20)
            index_score += (trend_score + position_score + change_score) * weight
        score += index_score * 0.4

        # 市场广度（权重30%）
        breadth = market_data['market_breadth']
        rise_ratio = breadth['rise_num'] / (breadth['rise_num'] + breadth['fall_num'])
        limit_ratio = breadth['limit_up'] / (breadth['limit_down'] + 1e-5)
        breadth_score = (min(rise_ratio * 100, 30)
                         + min(limit_ratio * 10, 20)
                         + (10 if breadth['limit_up'] > 50 else 0))

        score += breadth_score * 0.3

        # # 成交量（权重20%）
        # vol = market_data['volume']
        # vol_ratio = vol['current'] / vol['5d_avg']
        # vol_score = 20 if vol_ratio > 1.2 else 10 if vol_ratio > 0.8 else 0
        # score += vol_score * 0.2

        # 连板高度（权重10%）
        limit_stats = market_data['limit_stats']
        limit_score = (min(limit_stats['max_limit'] * 5, 15) + (10 if limit_stats.get(3, 0) > 5 else 0))
        score += limit_score * 0.3

        return max(min(score, 100), 0)

    def _calc_rise_score(self, ratio):
        """涨跌比评分（0-30分）"""
        if ratio >= 0.7:
            return 30
        elif ratio >= 0.6:
            return 25
        elif ratio >= 0.5:
            return 20
        else:
            return max(10 * ratio, 0)

    def _calc_limit_score(self, ratio):
        """涨跌停比评分（0-25分）"""
        if ratio >= 3: return 25
        if ratio >= 2: return 20
        if ratio >= 1: return 15
        return max(5 * ratio, 0)

    def _calc_turnover_score(self, data):
        """换手率动态评分（0-20分）"""
        # 计算换手率变化率
        current_turnover = data['avg_turnover']
        hist_mean = data['history']['turnover_rate'].mean()
        change_rate = (current_turnover - hist_mean) / hist_mean

        if change_rate > 0.2:  # 换手率上升20%以上
            return 20 if current_turnover > 5 else 15
        elif change_rate < -0.2:
            return 5
        else:
            return 10

    def _calc_volatility_score(self, data):
        """波动率评分（0-15分）"""
        if data['volatility'] < 1.5:  # 低波动
            return 5
        elif data['volatility'] < 3:  # 正常波动
            return 10
        else:  # 高波动
            return 15 if data['rise_num'] > data['fall_num'] else 5

    def _calc_limit_down_penalty(self, data):
        """跌停惩罚计算"""
        base_penalty = min(data['limit_down'] * 0.5, 20)  # 每个跌停扣0.5分，最多扣20分

        # 连续跌停惩罚
        hist_limit_down = data['history']['limit_down_count']
        if all(x > 30 for x in hist_limit_down[-3:]):  # 连续3日跌停>30
            base_penalty *= 1.5

        return min(base_penalty, 30)  # 总惩罚不超过30分

    def generate_trade_plan(self):
        """生成交易计划"""

        # 获取市场情绪数据
        market_data = self.get_market_sentiment()
        market_score = self.calculate_market_score(market_data)

        # 空值检查
        if not market_data or 'market_breadth' not in market_data:
            print("\n⚠️ 市场情绪数据获取失败")
            return []

        # 提取数据时使用安全访问方式
        market_breadth = market_data.get('market_breadth', {})
        rise_num = market_breadth.get('rise_num', 0)
        fall_num = market_breadth.get('fall_num', 0)
        limit_up = market_breadth.get('limit_up', 0)
        limit_down = market_breadth.get('limit_down', 0)

        # 打印信息时使用安全值
        print(f"\n市场情绪分析（{self.trade_date}）:")
        print(f"涨跌比: {rise_num}:{fall_num}")
        print(f"涨停/跌停: {limit_up}/{limit_down}")

        # 判断是否停止操作
        if market_score < self.market_params['sentiment_threshold']:
            print("\n⚠️ 市场情绪过差，建议空仓观望")
            return []

        zt_df = self.enhance_sector_analysis()
        if zt_df.empty:
            logger.warning("无有效涨停数据")
            return []

        # 计算综合评分
        zt_df['score'] = zt_df.apply(self.calculate_enhanced_score, axis=1)
        zt_df = zt_df.sort_values('score', ascending=False)

        # 在市场情绪分基础上调整个股得分
        adjust_ratio = 0.5 + market_score / 100 * 0.5  # 0.5-1.0的调整系数
        zt_df['score'] = zt_df['score'] * adjust_ratio

        # 过滤条件
        candidates = []
        for _, row in zt_df.iterrows():
            if row['reopen_count'] >= 2 or row['order_amount'] < 5e7:
                continue
            if row['score'] < 150:  # 阈值过滤
                continue

            candidates.append({
                'code': row['code'],
                'name': row['name'],
                'score': round(row['score'], 1),
                'price': round(row['close'] * 1.03, 2),
                'position': self._calculate_position(row),
                'reason': self._generate_reason(row)
            })
        return sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]

    def _calculate_position(self, row):
        """仓位管理模型"""
        base = 20
        # 封单金额加成（1亿加5%）
        base += min(row['order_amount'] / 1e8 * 5, 15)
        # 新增连板次数加成（每连板加3%仓位）
        limit_bonus = min(row.get('limit_count', 0) * 3, 15)
        base += limit_bonus
        # 时间加成
        if row['first_time'] <= '09:40:00':
            base += 15
        # 板块强度加成
        base += min(len(row['hot_sectors']) * 5, 10)
        return min(base, 50)

    def _generate_reason(self, row):
        """生成买入理由"""
        reasons = []
        if row['first_time'] <= '09:40:00':
            reasons.append('早盘封板')
        if row['order_amount'] >= 1e8:
            reasons.append(f"大单封板({row['order_amount'] / 1e8:.1f}亿)")
        if row['reopen_count'] == 0:
            reasons.append('零炸板')
        if len(row['hot_sectors']) > 0:
            reasons.append(f"热门板块:{','.join(row['hot_sectors'][:2])}")
            # 新增连板理由
        dt_score = self._calc_dragon_tiger_score(row['code'])
        if dt_score > 0:
            reasons.append(f"龙虎榜加分({dt_score})")
        elif dt_score < 0:
            reasons.append(f"龙虎榜减分({dt_score})")
        if 'limit_count' in row:
            try:
                count = int(row['limit_count'])
                if count >= 2:
                    reasons.append(f"{count}连板")
            except:
                pass
        return "，".join(reasons)

    def execute(self):
        """执行策略"""
        plan = self.generate_trade_plan()
        print(f"\n交易日{self.trade_date}竞价买入计划：")
        for idx, item in enumerate(plan, 1):
            print(f"{idx}. {item['name']}({item['code']}) 评分：{item['score']}")
            print(f"   报价：{item['price']} 仓位：{item['position']}%")
            print(f"   理由：{item['reason']}\n")


if __name__ == "__main__":
    strategy = PreOpenLimitUpStrategy()
    print(strategy.get_market_sentiment())