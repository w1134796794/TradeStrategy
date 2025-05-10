import akshare as ak
import pandas as pd
import logging
import psycopg2

from GetTradeDate import LocalTradeCalendar
from database import DatabaseManager
from datetime import datetime, date
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 新增数据库配置
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}


class DataFetcher:
    @staticmethod
    def fetch_zt_pool(date: str) -> pd.DataFrame:
        """获取涨停池数据"""
        try:
            return ak.stock_zt_pool_em(date)
        except Exception as e:
            logger.error(f"获取涨停池数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_zt_pool_hist(n_days: int = 5, end_date: str = None) -> pd.DataFrame:
        """获取最近N天的涨停池数据"""
        try:
            ltc = LocalTradeCalendar()
            if end_date is None:
                end_date = ltc.get_recent_trade_date()

            # 获取最近N天的交易日列表
            trade_dates = ltc.get_recent_trade_dates(end_date, n_days)

            result_df = pd.DataFrame()

            for date_str in trade_dates:
                try:
                    daily_df = DataFetcher.fetch_zt_pool(date=date_str)
                    if not daily_df.empty:
                        daily_df["日期"] = date_str  # 添加日期列
                        result_df = pd.concat([result_df, daily_df], ignore_index=True)
                except Exception as e:
                    logger.warning(f"获取{date_str}涨停数据失败: {str(e)}")

            return result_df
        except Exception as e:
            logger.error(f"获取最近{str(n_days)}天涨停池数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_sector_list(sector_type: str) -> pd.DataFrame:
        """获取板块列表"""
        try:
            if sector_type == "industry":
                df = ak.stock_board_industry_name_em()
                df = df.rename(columns={"板块代码": "板块代码", "板块名称": "板块名称"})
                return df
            elif sector_type == "concept":
                df = ak.stock_board_concept_name_em()
                df = df.rename(columns={"代码": "板块代码", "名称": "板块名称"})
                df["板块名称"] = df["板块名称"].str.replace(r"\s+", "", regex=True)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取{sector_type}板块列表失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_sector_history(sector: str, sector_type: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取板块历史数据"""
        try:
            if sector_type == "industry":
                return ak.stock_board_industry_hist_em(
                    symbol=sector, period="日k",
                    start_date=start_date, end_date=end_date
                )
            elif sector_type == "concept":
                return ak.stock_board_concept_hist_em(
                    symbol=sector, period="daily",
                    start_date=start_date, end_date=end_date
                )
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取{sector_type}板块历史数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_lhb_detail(start_date: str, end_date: str) -> pd.DataFrame:
        """获取龙虎榜主表数据"""
        try:
            df = ak.stock_lhb_detail_em(start_date=start_date, end_date=end_date)
            if not df.empty:
                df = df[df['代码'].str.contains('^(0|3|4|6|8|9)')]
                df['上榜后1日'] = df['上榜后1日'].fillna(0)
                df['上榜后2日'] = df['上榜后2日'].fillna(0)
                df['上榜后5日'] = df['上榜后5日'].fillna(0)
                df['上榜后10日'] = df['上榜后10日'].fillna(0)
                return df.rename(columns={"代码": "股票代码", "名称": "股票名称", "上榜日": "数据日期"})
            return df
        except Exception as e:
            logger.error(f"龙虎榜主表获取失败 {start_date}-{end_date}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_lhb_stock_detail(code: str, date: str, flag: str) -> pd.DataFrame:
        """获取个股龙虎榜明细"""
        try:
            df = ak.stock_lhb_stock_detail_em(symbol=code, date=date, flag=flag)
            if not df.empty:
                df["股票代码"] = code
                df["数据日期"] = date
                return df
            return df
        except Exception as e:
            logger.error(f"获取个股龙虎榜明细失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_index_data(index_code: str) -> pd.DataFrame:
        """获取指数数据"""
        try:
            return ak.stock_zh_index_daily(symbol=index_code)
        except Exception as e:
            logger.error(f"获取指数{index_code}数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_market_breadth() -> pd.DataFrame:
        """获取市场广度数据"""
        try:
            return ak.stock_market_activity_legu()
        except Exception as e:
            logger.error(f"获取市场广度数据失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_stock_hist(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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

    @staticmethod
    def fetch_stock_info() -> pd.DataFrame:
        """获取股票基本信息"""
        try:
            return ak.stock_info_a_code_name()
        except Exception as e:
            logger.error(f"获取股票信息失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_market_volume(market_type: str) -> pd.DataFrame:
        """获取市场成交数据"""
        market_map = {
            "sh": ak.stock_sh_a_spot_em,
            "sz": ak.stock_sz_a_spot_em,
            "bj": ak.stock_bj_a_spot_em
        }
        try:
            return market_map[market_type]()
        except Exception as e:
            logger.error(f"获取{market_type}市场数据失败: {str(e)}")
            return pd.DataFrame()


class DataSaver:

    @staticmethod
    def save_to_db(df: pd.DataFrame, table: str, conflict_keys: List[str]):
        """保存到数据库"""
        if df.empty:
            logging.warning(f"空数据，跳过保存到表 {table}")
            return

        db = DatabaseManager(**DB_CONFIG)
        try:
            columns = df.columns.tolist()
            data = [tuple(row) for row in df.itertuples(index=False)]
            db.upsert_data(table, columns, data, conflict_keys)
            logging.info(f"成功保存 {len(data)} 行到表 {table}")
        finally:
            db.close()

    @staticmethod
    def execute_query(query, params=None):
        """执行查询并返回DataFrame"""
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    data = cursor.fetchall()
                    return pd.DataFrame(data, columns=columns)
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"数据库查询失败: {str(e)}")
            return pd.DataFrame()


class DataProcessor:
    @staticmethod
    def process(today: str):
        """数据处理主流程"""
        logger.info("开始执行数据处理流程...")
        ltc = LocalTradeCalendar()

        # 获取并保存各类数据
        # -------save to db -------------
        # DataProcessor._save_zt_pool_new(today)
        # DataProcessor._save_sector_list_new()
        # DataProcessor._save_sector_history_new(today)
        # DataProcessor._save_lhb_data_new(today)
        # DataProcessor._save_index_data_new()
        # DataProcessor._save_market_volume_new(today)
        DataProcessor._save_sector_components_new()
        DataProcessor._save_daily_stock_new(today)
        DataProcessor._save_fund_rank_new(today)
        DataProcessor._save_fund_bj_new(today)

    @staticmethod
    def _save_zt_pool_new(today: str):
        """保存涨停池数据"""
        zt_df = DataFetcher.fetch_zt_pool(today)
        if not zt_df.empty:
            zt_df['日期'] = today
            zt_df['涨跌幅'] = round(zt_df['涨跌幅'], 2)
            DataSaver.save_to_db(zt_df, 'zt_pool_hist', ['日期', '代码'])

    @staticmethod
    def _save_sector_list_new():
        """保存板块列表数据"""
        sector_list_data = pd.DataFrame()

        for sector_type in ["industry", "concept"]:
            sector_list = DataFetcher.fetch_sector_list(sector_type)
            if not sector_list.empty:
                sector_list["板块类型"] = sector_type
                sector_list_data = pd.concat([sector_list_data, sector_list], ignore_index=True)

        if not sector_list_data.empty:
            DataSaver.save_to_db(sector_list_data, 'sector_list', ['板块代码'])

    @staticmethod
    def _save_sector_history_new(today: str):
        """保存板块历史数据"""
        sector_hist_data = pd.DataFrame()

        for sector_type in ["industry", "concept"]:
            sector_list = DataFetcher.fetch_sector_list(sector_type)
            if not sector_list.empty:
                for _, row in sector_list.iterrows():
                    hist_df = DataFetcher.fetch_sector_history(
                        sector=row["板块名称"],
                        sector_type=sector_type,
                        start_date=today,
                        end_date=today
                    )
                    if not hist_df.empty:
                        hist_df["板块代码"] = row["板块代码"]
                        hist_df["板块名称"] = row["板块名称"]
                        hist_df["板块类型"] = sector_type
                        sector_hist_data = pd.concat([sector_hist_data, hist_df], ignore_index=True)

        if not sector_hist_data.empty:
            DataSaver.save_to_db(sector_hist_data, 'sector_hist', ['日期', '板块代码'])

    @staticmethod
    def _save_lhb_data_new(today: str):
        """保存龙虎榜数据"""
        # 主表
        lhb_main = DataFetcher.fetch_lhb_detail(today, today)
        if lhb_main.empty:
            logger.warning("龙虎榜主表数据为空")
            return

        lhb_main.sort_values(by=["股票代码", "股票名称"], ascending=True, inplace=True)
        DataSaver.save_to_db(lhb_main, 'lhb_main', ['数据日期', '股票代码', '上榜原因'])

        # 明细表
        if not lhb_main.empty:
            lhb_details = []
            for _, row in lhb_main.iterrows():
                # 确保日期是字符串格式
                date_str = row["数据日期"].strftime("%Y%m%d") if isinstance(row["数据日期"], (date, datetime)) else row[
                    "数据日期"]
                for flag in ["买入", "卖出"]:
                    detail_df = DataFetcher.fetch_lhb_stock_detail(
                        code=row["股票代码"],
                        date=date_str,
                        flag=flag
                    )
                    if not detail_df.empty:
                        detail_df["方向"] = flag
                        # 重新排列列顺序
                        detail_df = detail_df.reindex(columns=[
                            "序号", "股票代码", "数据日期", "方向", "交易营业部名称",
                            "买入金额", "买入金额-占总成交比例", "卖出金额", "卖出金额-占总成交比例",
                            "净额", "类型"
                        ])
                        lhb_details.append(detail_df)
                    else:
                        logger.warning(f"获取个股龙虎榜明细失败: {row['股票代码']} - {date_str} - {flag}")

            if lhb_details:
                combined_df = pd.concat(lhb_details, ignore_index=True)
                combined_df.sort_values(by=["股票代码", "数据日期", "方向"], ascending=True, inplace=True)
                DataSaver.save_to_db(combined_df, 'lhb_detail', ['数据日期', '股票代码', '方向', '序号'])

    @staticmethod
    def _save_index_data_new():
        """保存指数数据"""
        index_codes = ["sh000001", "sz399001", "sz399006", "sh000016"]
        index_data = pd.DataFrame()

        for code in index_codes:
            index_df = DataFetcher.fetch_index_data(code)
            if not index_df.empty:
                index_df['index_code'] = code
                index_df['date'] = pd.to_datetime(index_df['date']).dt.strftime('%Y%m%d')
                index_data = pd.concat([index_data, index_df], ignore_index=True)

        if not index_data.empty:
            DataSaver.save_to_db(index_data, 'index_data', ['index_code', 'date'])

    @staticmethod
    def _save_market_volume_new(today: str):
        """保存市场成交量数据"""
        market_volume_data = pd.DataFrame()

        for market in ["sh", "sz", "bj"]:
            volume_df = DataFetcher.fetch_market_volume(market)
            if not volume_df.empty:
                volume_df["市场类型"] = market
                volume_df['日期'] = today
                market_volume_data = pd.concat([market_volume_data, volume_df], ignore_index=True)

        # 新增数据清洗步骤
        if not market_volume_data.empty:
            # 删除最新价为NaN的行
            clean_data = market_volume_data.dropna(subset=['最新价'])

            # 添加清洗结果日志
            removed_count = len(market_volume_data) - len(clean_data)
            logging.info(f"数据清洗：删除{removed_count}条无效最新价记录")

            # 保存有效数据
            if not clean_data.empty:
                DataSaver.save_to_db(clean_data, 'market_volume', ['日期', '代码'])
                logging.info(f"成功保存{len(clean_data)}条有效数据")
            else:
                logging.warning("清洗后无有效数据可保存")
        else:
            logging.warning("未获取到任何市场数据")

    @staticmethod
    def _save_sector_components_new():
        """保存板块成分股数据"""
        sector_components_data = pd.DataFrame()

        # 获取行业板块列表
        industry_list = DataFetcher.fetch_sector_list("industry")
        if not industry_list.empty:
            for _, row in industry_list.iterrows():
                # 获取行业板块成分股
                components_df = ak.stock_board_industry_cons_em(symbol=row["板块名称"])
                if not components_df.empty:
                    components_df["板块代码"] = row["板块代码"]
                    components_df["板块名称"] = row["板块名称"]
                    components_df["板块类型"] = "industry"
                    sector_components_data = pd.concat([sector_components_data, components_df], ignore_index=True)

        # 获取概念板块列表
        concept_list = DataFetcher.fetch_sector_list("concept")
        if not concept_list.empty:
            for _, row in concept_list.iterrows():
                # 获取概念板块成分股
                components_df = ak.stock_board_concept_cons_em(symbol=row["板块名称"])
                if not components_df.empty:
                    components_df["板块代码"] = row["板块代码"]
                    components_df["板块名称"] = row["板块名称"]
                    components_df["板块类型"] = "concept"
                    sector_components_data = pd.concat([sector_components_data, components_df], ignore_index=True)

        if not sector_components_data.empty:
            sector_components_data = sector_components_data.drop_duplicates(subset=['代码', '板块代码'])
            DataSaver.save_to_db(sector_components_data, 'sector_components', ['代码', '板块代码'])

    @staticmethod
    def _save_daily_stock_new(today: str):
        """保存市场成交量数据"""
        import tushare as ts
        pro = ts.pro_api('783cbe265c538fc2d6d7fd11d966d9cf0757ac70bec9f8f66cec9e8b')

        df_new = pro.daily(trade_date=today)

        if not df_new.empty:
            # 处理股票代码，去掉后缀只保留6位数字
            df_new['ts_code'] = df_new['ts_code'].apply(lambda x: x.split('.')[0])
            df_new.rename(columns={
                'trade_date': '日期',
                'ts_code': '股票代码',
                'open': '开盘',
                'close': '收盘',
                'pre_close': '昨收',
                'high': '最高',
                'low': '最低',
                'vol': '成交量',
                'amount': '成交额',
                'pct_chg': '涨跌幅',
                'change': '涨跌额'
            }, inplace=True)

            # 成交额转为元
            df_new['成交额'] = df_new['成交额'] * 1000

            DataSaver.save_to_db(df_new, 'daily_stock_prices', ['日期', '股票代码'])

    @staticmethod
    def _save_fund_rank_new(today):
        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        df['日期'] = today

        correct_columns = [
            '序号', '日期', '代码', '名称', '最新价', '今日涨跌幅',
            '今日主力净流入净额', '今日主力净流入净占比',
            '今日超大单净流入净额', '今日超大单净流入净占比',
            '今日大单净流入净额', '今日大单净流入净占比',
            '今日中单净流入净额', '今日中单净流入净占比',
            '今日小单净流入净额', '今日小单净流入净占比'
        ]

        df = df.rename(columns={
            '今日主力净流入-净额': '今日主力净流入净额',
            '今日主力净流入-净占比': '今日主力净流入净占比',
            '今日超大单净流入-净额': '今日超大单净流入净额',
            '今日超大单净流入-净占比': '今日超大单净流入净占比',
            '今日大单净流入-净额': '今日大单净流入净额',
            '今日大单净流入-净占比': '今日大单净流入净占比',
            '今日中单净流入-净额': '今日中单净流入净额',
            '今日中单净流入-净占比': '今日中单净流入净占比',
            '今日小单净流入-净额': '今日小单净流入净额',
            '今日小单净流入-净占比': '今日小单净流入净占比'
            })
        df = df.reindex(columns=correct_columns)
        df = df.drop(df[(df[correct_columns] == '-').any(axis=1)].index)

        DataSaver.save_to_db(df, 'fund_flow', ['日期', '代码'])

    @staticmethod
    def _save_fund_bj_new(today: str):
        """获取当日北京市场股票资金流入"""
        query = """
            SELECT 代码, 名称, 日期, 市场类型, 最新价
            FROM market_volume 
            WHERE 市场类型 = 'bj' 
              AND 日期 = %s
        """
        df_stocks = DataSaver.execute_query(query=query, params=(today,))

        if df_stocks.empty:
            logging.warning(f"{today} 无北京市场股票数据")
            return

        all_data = []
        for _, row in df_stocks.iterrows():
            try:
                # 获取资金流数据
                df_flow = ak.stock_individual_fund_flow(stock=row['代码'], market="bj")

                ##############################################
                # 增强数据校验逻辑
                ##############################################
                # 1. 检查数据是否存在
                if df_flow is None:
                    logging.warning(f"股票 {row['代码']} 返回None数据")
                    continue

                # 2. 检查数据结构是否有效
                required_columns = {'日期', '收盘价', '涨跌幅', '主力净流入-净额'}
                if not required_columns.issubset(df_flow.columns):
                    missing_cols = required_columns - set(df_flow.columns)
                    logging.warning(f"股票 {row['代码']} 缺少必要字段: {missing_cols}")
                    continue

                # 3. 日期格式处理
                df_flow['日期'] = pd.to_datetime(df_flow['日期'], errors='coerce').dt.strftime('%Y%m%d')
                if df_flow['日期'].isnull().all():
                    logging.warning(f"股票 {row['代码']} 日期格式无效")
                    continue

                # 4. 筛选当日数据
                df_flow = df_flow[df_flow['日期'] == today]
                if df_flow.empty:
                    logging.info(f"股票 {row['代码']} 无当日数据")
                    continue

                # 5. 添加标识信息
                df_flow = df_flow.assign(代码=row['代码'], 名称=row['名称'])
                all_data.append(df_flow)

            except KeyError as e:
                logging.error(f"数据结构异常 {row['代码']}: {str(e)}")
            except TypeError as e:
                logging.error(f"类型错误 {row['代码']}: {str(e)}")
            except Exception as e:
                logging.error(f"处理股票 {row['代码']} 失败: {str(e)}", exc_info=True)

        # 数据合并与清洗
        if not all_data:
            logging.warning("无有效数据可保存")
            return

        final_df = pd.concat(all_data, ignore_index=True)

        # 列名映射与清洗
        column_mapping = {
            '收盘价': '最新价',
            '涨跌幅': '今日涨跌幅',
            '主力净流入-净额': '今日主力净流入净额',
            '主力净流入-净占比': '今日主力净流入净占比',
            '超大单净流入-净额': '今日超大单净流入净额',
            '超大单净流入-净占比': '今日超大单净流入净占比',
            '大单净流入-净额': '今日大单净流入净额',
            '大单净流入-净占比': '今日大单净流入净占比',
            '中单净流入-净额': '今日中单净流入净额',
            '中单净流入-净占比': '今日中单净流入净占比',
            '小单净流入-净额': '今日小单净流入净额',
            '小单净流入-净占比': '今日小单净流入净占比'
        }

        # 仅映射实际存在的列
        final_df = final_df.rename(columns={
            k: v for k, v in column_mapping.items()
            if k in final_df.columns
        })

        # 关键字段校验
        mandatory_fields = ['代码', '日期', '最新价']
        if not set(mandatory_fields).issubset(final_df.columns):
            logging.error("合并后数据缺少关键字段")
            return

        # 数据保存
        try:
            if not final_df.empty:
                DataSaver.save_to_db(
                    df=final_df,
                    table='fund_flow',
                    conflict_keys=['日期', '代码']
                )
                logging.info(f"成功保存 {len(final_df)} 条记录")
            else:
                logging.warning("清洗后无有效数据")
        except Exception as e:
            logging.error("数据库保存失败", exc_info=True)


def main():
    ltc = LocalTradeCalendar()
    today = ltc.get_recent_trade_date()
    DataProcessor.process(today)
    logger.info(f"\n=== 数据存储完成 ===")


if __name__ == "__main__":
    main()
