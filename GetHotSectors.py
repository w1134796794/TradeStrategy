import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Set
import logging
from GetTradeDate import TradeCalendar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)

class SectorAnalyzer:
    """热门板块分析器（优化版）"""

    def __init__(self, trade_date: str = None):
        """
        初始化分析器
        :param trade_date: 交易日（格式：YYYYMMDD），默认取最近交易日
        """
        self.calender = TradeCalendar()
        self.trade_date = trade_date or self._get_recent_trade_date()
        self.sector_stocks_map: Dict[Tuple[str, str], Set[str]] = {}
        self.hot_sectors: List[Tuple[str, str]] = []
        self.sector_type_map = {}
        self.config = {
            'days': 3,  # 评估周期
            'index_code': 'sh000001',  # 大盘指数代码
            'net_inflow_threshold': 5e7,  # 大单净流入阈值(元)
            'fast_limit_hour': 10,  # 快速涨停时间阈值(小时)
            'weights': {  # 指标权重
                'recent_gain': 0.5,
                'resist_market': 0.3,
                'fast_limit': 0.2
            }
        }

    def _get_start_date(self, days: int) -> str:
        """获取days个交易日前的日期"""
        return self.calender.get_previous_trade_date(
            base_date=self.trade_date,
            days=days
        )

    def _get_recent_trade_date(self) -> str:
        """获取最近交易日（需根据实际交易日历实现）"""
        return self.calender.get_recent_trade_date()

    def _get_sector_hist_data(self, sector: str, sector_type: str, start_date: str, end_date: str) -> pd.DataFrame:
        """统一处理不同板块的历史数据获取"""
        try:
            # 定义不同板块的参数模板
            params_config = {
                "industry": {
                    "hist_func": ak.stock_board_industry_hist_em,
                    "period": "日k",
                    "param_order": ["symbol", "start_date", "end_date", "period", "adjust"]
                },
                "concept": {
                    "hist_func": ak.stock_board_concept_hist_em,
                    "period": "daily",
                    "param_order": ["symbol", "period", "start_date", "end_date", "adjust"]
                }
            }

            config = params_config[sector_type]
            args = {
                "symbol": sector,
                "start_date": start_date,
                "end_date": end_date,
                "period": config["period"],
                "adjust": ""
            }

            # 按照接口要求的参数顺序构建参数列表
            ordered_args = [args[key] for key in config["param_order"]]

            # 动态调用接口
            return config["hist_func"](*ordered_args)

        except KeyError as e:
            logger.error(f"不支持的板块类型: {sector_type}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取{sector_type}板块[{sector}]数据失败: {str(e)}")
            return pd.DataFrame()

    def _get_sector_ranking(self, sector_type: str, days: int) -> List[dict]:
        """通用板块排行获取方法"""
        try:
            # 获取板块列表（新增名称清洗）
            if sector_type == "industry":
                name_df = ak.stock_board_industry_name_em()
            elif sector_type == "concept":
                name_df = ak.stock_board_concept_name_em()
                name_df["板块名称"] = name_df["板块名称"].str.replace(" ", "")
            else:
                return []

            sector_list = name_df["板块名称"].tolist()
            trade_date_obj = datetime.strptime(self.trade_date, "%Y%m%d")
            start_date = (trade_date_obj - timedelta(days=days + 2)).strftime("%Y%m%d")

            rankings = []
            for sector in sector_list:
                try:
                    # 使用统一接口获取数据
                    hist_df = self._get_sector_hist_data(
                        sector=sector,
                        sector_type=sector_type,
                        start_date=start_date,
                        end_date=self.trade_date
                    )

                    # 数据有效性检查
                    if hist_df.empty or "涨跌幅" not in hist_df.columns:
                        logger.warning(f"板块[{sector}]数据无效，跳过")
                        continue

                    # 计算分数时处理可能的空值
                    last_days = hist_df["涨跌幅"].tail(days).replace(np.nan, 0)

                    rankings.append({
                        "name": sector,
                        "type": sector_type,
                        "score": round(last_days.sum(), 4),
                        "data_points": len(last_days)
                    })

                except Exception as e:
                    logger.warning(f"处理板块[{sector}]时发生错误: {str(e)}")
                    continue

            return rankings

        except Exception as e:
            logger.error(f"获取{sector_type}板块列表失败: {str(e)}")
            return []

    def get_hot_sectors(self, days: int = 2, top_n_per_type: int = 5) -> List[Tuple[str, str, float]]:
        """获取热门板块排行（行业+概念各取前N）"""
        try:
            # 并行获取行业和概念数据
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_industry = executor.submit(self._get_sector_ranking, "industry", days)
                future_concept = executor.submit(self._get_sector_ranking, "concept", days)

                industry_rank = future_industry.result()
                concept_rank = future_concept.result()

            # 分别处理行业和概念排行
            def process_ranking(ranking: List[dict], sector_type: str) -> pd.DataFrame:
                df = pd.DataFrame(ranking)
                if df.empty:
                    return df
                return df.nlargest(top_n_per_type, 'score', keep='first')

            # 获取行业前5
            industry_top = process_ranking(industry_rank, "industry")
            # 获取概念前5
            concept_top = process_ranking(concept_rank, "concept")

            # 合并结果
            combined = pd.concat([industry_top, concept_top])
            if combined.empty:
                logger.warning("未获取到有效板块数据")
                return []

            # 按分数降序排列
            final_df = combined.sort_values('score', ascending=False)
            self.hot_sectors = [
                (row['name'], row['type'], float(row['score']))
                for _, row in final_df.iterrows()
            ]
            # 动态填充 sector_type_map
            for name, sector_type, _ in self.hot_sectors:
                self.sector_type_map[name] = sector_type
            return self.hot_sectors

        except Exception as e:
            logger.error(f"获取热门板块失败: {str(e)}", exc_info=True)
            return []

    def _get_sector_components(self, sector_name: str, sector_type: str) -> Set[str]:
        """获取板块成分股（增强版）"""
        try:
            # 清洗板块名称（去除空格和特殊字符）
            cleaned_name = sector_name.strip().replace(" ", "").replace("　", "")  # 处理全角空格

            # 根据板块类型调用接口
            if sector_type == "industry":
                df = ak.stock_board_industry_cons_em(symbol=cleaned_name)
            elif sector_type == "concept":
                df = ak.stock_board_concept_cons_em(symbol=cleaned_name)
            else:
                logger.warning(f"未知板块类型: {sector_type}")
                return set()

            # 验证数据有效性
            if df.empty:
                logger.warning(f"板块[{cleaned_name}]返回空数据")
                return set()

            if '代码' not in df.columns:
                logger.error(f"板块[{cleaned_name}]数据字段异常，实际字段: {df.columns.tolist()}")
                return set()

            # 统一代码格式并去重
            codes = df['代码'].astype(str).str.zfill(6).unique().tolist()
            logger.info(f"获取板块[{cleaned_name}]成分股成功，共{len(codes)}只股票")
            return set(codes)

        except Exception as e:
            logger.error(f"获取[{sector_type}]板块[{sector_name}]成分股异常: {str(e)}", exc_info=True)
            return set()

    def build_sector_map(self, hot_sectors: List, max_workers: int = 5) -> Dict:
        """构建板块成分股映射（增强版）"""
        if not hot_sectors:
            logger.warning("无热门板块数据")
            return {}

        sector_map = {}
        failed_sectors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 修改点：遍历时解包三个参数，忽略第三个得分参数
            futures = {
                executor.submit(self._get_sector_components, name, sector_type): (name, sector_type)
                for name, sector_type, _ in hot_sectors  # 关键修复：添加 _, 解包三个参数
            }

            # 处理完成的任务
            for future in as_completed(futures):
                sector_name, sector_type = futures[future]
                try:
                    codes = future.result()
                    if codes:
                        sector_map[(sector_name, sector_type)] = codes
                        logger.debug(f"成功处理板块[{sector_name}]，获取{len(codes)}成分股")
                    else:
                        failed_sectors.append((sector_name, sector_type))
                        logger.warning(f"板块[{sector_name}]未获取到有效成分股")
                except Exception as e:
                    failed_sectors.append((sector_name, sector_type))
                    logger.error(f"处理板块[{sector_name}]时发生未捕获异常: {str(e)}")

        # 记录最终结果
        logger.info(f"板块映射构建完成，成功{len(sector_map)}个，失败{len(failed_sectors)}个")
        if failed_sectors:
            logger.warning(f"失败板块列表: {failed_sectors}")

        self.sector_stocks_map = sector_map
        return sector_map

    def enhance_analysis(self, zt_df: pd.DataFrame) -> pd.DataFrame:
        """增强涨停板数据分析"""
        if not isinstance(zt_df, pd.DataFrame) or zt_df.empty:
            logger.warning("输入的涨停板数据为空")
            return pd.DataFrame()

        if not self.sector_stocks_map:
            logger.warning("请先调用build_sector_map构建板块映射")
            return pd.DataFrame()

        # 预处理：统一代码格式
        zt_df['code'] = zt_df['code'].astype(str).str.zfill(6)

        # 构建代码到板块的映射
        code_to_sectors = {}
        for (sector_name, sector_type), codes in self.sector_stocks_map.items():
            for code in codes:
                if code not in code_to_sectors:
                    code_to_sectors[code] = []
                code_to_sectors[code].append(f"{sector_name}({sector_type})")

        # 批量匹配
        zt_df['hot_sectors'] = zt_df['code'].map(code_to_sectors).apply(
            lambda x: x if isinstance(x, list) else []
        )

        return zt_df

    def get_sector_momentum(self, sector: str, sector_type: str, days: int = 3) -> float:
        """计算板块动量指标（修正日期问题）"""
        try:
            # 获取历史数据
            start_date = self._get_start_date(days)
            hist_df = self._get_sector_hist_data(
                sector=sector,
                sector_type=sector_type,
                start_date=start_date,
                end_date=self.trade_date
            )

            # 列名标准化处理
            column_map = {
                '收盘': 'close',
                '收盘价': 'close',
                '成交量': 'volume',
                '主力净流入': 'fund_flow'
            }
            hist_df = hist_df.rename(columns=column_map)

            # 动量计算
            momentum_score = 0

            # 价格动量（40%）
            if 'close' in hist_df.columns:
                price_change = hist_df['close'].pct_change(days).iloc[-1] * 100
                momentum_score += min(abs(price_change) * 0.4, 40)

            # 成交量动量（30%）
            if 'volume' in hist_df.columns:
                vol_change = hist_df['volume'].pct_change(days).iloc[-1] * 100
                momentum_score += min(abs(vol_change) * 0.3, 30)

            # 资金流向（30%）
            if 'fund_flow' in hist_df.columns:
                fund_flow = hist_df['fund_flow'].sum()
                momentum_score += min(fund_flow / 1e8, 30)  # 每亿加1分

            return min(momentum_score, 100)

        except Exception as e:
            logger.error(f"板块[{sector}]动量计算失败: {str(e)}")
            return 0

    def get_sector_dragons(self, sector: str, sector_type: str) -> List[Dict]:
        """获取板块龙一到龙五"""
        # 获取成分股
        components = self._get_sector_components(sector, sector_type)

        # 获取个股多维数据
        stock_data = self._collect_stock_data(list(components)[:50])  # 限制前50只

        # 获取大盘数据
        index_data = ak.stock_zh_index_daily(symbol=self.config['index_code']).iloc[-10:]

        # 计算评分
        scored_stocks = []
        for code, data in stock_data.items():
            score = self._calculate_stock_score(data, index_data)
            scored_stocks.append({**data, **score})

        # 排序返回前五
        df = pd.DataFrame(scored_stocks)
        if df.empty:
            return []

        return df.sort_values('total_score', ascending=False).head(5).to_dict('records')

    def _collect_stock_data(self, codes: List[str]) -> Dict:
        """批量获取股票多维数据"""
        stock_data = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(self._get_single_data, code): code for code in codes}
            for future in as_completed(futures):
                code = futures[future]
                try:
                    data = future.result(timeout=30)
                    if data:
                        stock_data[code] = data
                except Exception as e:
                    logger.error(f"股票{code}数据获取失败: {str(e)}")
        return stock_data

    def _get_single_data(self, code: str) -> Dict:
        """获取单只股票数据（增强类型检查和异常处理）"""
        try:
            # ==== 获取全市场实时数据 ====
            spot_df = ak.stock_zh_a_spot_em()
            stock_spot = spot_df[spot_df['代码'] == code]
            if stock_spot.empty:
                logger.warning(f"股票{code}未找到实时数据")
                return None

            # ==== 处理市值字段（兼容不同类型）====
            market_cap_value = stock_spot['总市值'].iloc[0]
            if isinstance(market_cap_value, str):
                # 处理字符串格式（如"100.5亿"）
                market_cap = float(market_cap_value.replace('亿', '')) * 1e8
            elif isinstance(market_cap_value, (float, int)):
                # 处理数值格式（单位已经是亿元）
                market_cap = float(market_cap_value) * 1e8
            else:
                logger.warning(f"股票{code}市值格式异常: {type(market_cap_value)}")
                market_cap = 0.0

            # ==== 获取历史数据 ====
            start_date = self.calender.get_previous_trade_date(
                base_date=self.trade_date,
                days=self.config['days']
            )
            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                adjust="hfq",
                start_date=start_date,
                end_date=self.trade_date
            )
            if len(hist) < self.config['days']:
                logger.warning(f"股票{code}历史数据不足{self.config['days']}天")
                return None

            recent_gain = (hist['收盘'].iloc[-1] / hist['收盘'].iloc[0] - 1) * 100

            # ==== 涨停数据 ====
            limit_time = None
            try:
                zt_data = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
                zt_stock = zt_data[zt_data['代码'] == code]
                if not zt_stock.empty and '首次封板时间' in zt_stock.columns:
                    raw_time = str(zt_stock['首次封板时间'].iloc[0])
                    try:
                        # 补全为6位数字
                        time_str = raw_time.zfill(6)

                        # 格式化为HH:MM:SS
                        formatted_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"

                        # 拼接完整日期
                        current_date = datetime.now().strftime("%Y-%m-%d")
                        full_datetime = f"{current_date} {formatted_time}"

                        # 解析为datetime对象
                        limit_time = pd.to_datetime(full_datetime, format='%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        logger.error(f"解析涨停时间失败: {raw_time} → 错误: {str(e)}")
                        limit_time = None
            except Exception as e:
                logger.error(f"获取股票{code}涨停数据失败: {str(e)}")

            return {
                'code': code,
                'name': stock_spot['名称'].iloc[0],
                'recent_gain': recent_gain,
                'limit_time': limit_time,
                'hist_data': hist,
                'market_cap': market_cap
            }
        except Exception as e:
            logger.error(f"股票{code}数据处理失败: {str(e)}", exc_info=True)
            return None

    def _calculate_stock_score(self, data: Dict, index_data: pd.DataFrame) -> Dict:
        """股票评分计算（全部分类独立百分制）"""
        # 近期涨幅得分（0-100分）
        gain_score = self._normalize(
            value=data['recent_gain'],
            min_val=5,  # 假设5%涨幅为基准值
            max_val=30  # 30%涨幅为满分阈值
        ) * 100

        # 逆势得分（0-100分）
        resist_score = min(self._calculate_resist_score(
            stock_data=data['hist_data'],
            index_data=index_data
        ), 100)  # 确保不超过100

        # 封板速度得分（0或100分）
        fast_limit_score = 100 if (
                data['limit_time'] and
                data['limit_time'].hour < self.config['fast_limit_hour']
        ) else 0

        # 加权总分（权重和为1时总分自然0-100）
        total_score = (
                gain_score * self.config['weights']['recent_gain'] +
                resist_score * self.config['weights']['resist_market'] +
                fast_limit_score * self.config['weights']['fast_limit']
        )

        return {
            'gain_score': round(gain_score, 1),  # 近期涨幅（0-100）
            'resist_score': round(resist_score, 1),  # 逆势能力（0-100）
            'fast_limit_score': fast_limit_score,  # 封板速度（0/100）
            'total_score': round(total_score, 1)  # 综合得分（0-100）
        }

    def _calculate_resist_score(self, stock_data: pd.DataFrame, index_data: pd.DataFrame) -> float:
        """计算逆势得分（完整字段映射版本）"""
        try:
            # ===== 1. 字段映射预处理 =====
            # 股票数据列名标准化
            stock_df = stock_data.rename(columns={
                "日期": "date",  # 日期字段统一
                "涨跌幅": "pct_change",  # 涨跌幅字段映射
                "收盘": "close"  # 收盘价映射（可选）
            })

            # 指数数据列名标准化
            index_df = index_data.rename(columns={
                "close": "index_close"  # 明确指数收盘价
            }).copy()

            # ===== 2. 计算指数涨跌幅 =====
            # 确保数据按日期排序
            index_df = index_df.sort_values("date")
            # 计算日涨跌幅（保留原始收盘价）
            index_df["index_pct_change"] = index_df["index_close"].pct_change() * 100
            index_df = index_df.dropna(subset=["index_pct_change"])

            # ===== 3. 数据合并 =====
            # 转换日期格式（确保两边都是datetime）
            stock_df["date"] = pd.to_datetime(stock_df["date"])
            index_df["date"] = pd.to_datetime(index_df["date"])

            # 合并数据（使用inner join确保日期对齐）
            merged_df = pd.merge(
                stock_df[["date", "pct_change"]],  # 只需要日期和股票涨跌幅
                index_df[["date", "index_pct_change"]],
                on="date",
                how="inner"
            )

            # ===== 4. 计算逆势天数 =====
            if merged_df.empty:
                logger.warning("股票与指数数据无共同交易日")
                return 0.0

            # 计算条件：指数下跌且股票上涨
            resist_condition = ((merged_df["index_pct_change"] < 0) & (merged_df["pct_change"] > 0))

            resist_days = resist_condition.sum()
            total_days = len(merged_df)

            # ===== 5. 最终得分计算 =====
            resist_score = (resist_days / total_days) * 100 if total_days > 0 else 0
            return min(resist_score, 100)  # 确保不超过100分

        except Exception as e:
            logger.error(f"逆势得分计算失败: {str(e)}", exc_info=True)
            return 0.0

    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """标准化到0-1范围（自动处理极端值）"""
        if max_val <= min_val:
            return 0.0
        clipped_value = max(min(value, max_val), min_val)
        return (clipped_value - min_val) / (max_val - min_val)


# 使用示例
if __name__ == "__main__":
    analyzer = SectorAnalyzer(trade_date="20250328")

    # 获取近2日热门板块前5
    hot_sectors = analyzer.get_hot_sectors(days=2, top_n_per_type=1)
    print("热门板块:", hot_sectors)

    # 构建成分股映射
    sector_map = analyzer.build_sector_map(hot_sectors)

    for sector_name, sector_type, _ in hot_sectors:
        dragons = analyzer.get_sector_dragons(sector_name, sector_type)
        print(f"\n【{sector_name}】龙头榜单：")
        for i, stock in enumerate(dragons, 1):
            print(
                f"龙{i} {stock.get('name', '未知')}({stock.get('code', '未知')}) "
                f"得分：{stock.get('total_score', 0):.1f}\n"
                f"  近期涨幅：{stock.get('recent_gain', 0):.1f}% "
                f"逆势得分：{stock.get('resist_score', 0):.1f} "
                f"快速上板得分：{stock.get('limit_time', 0):.1f} "
            )