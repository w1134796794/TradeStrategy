import akshare as ak
import pandas as pd
import logging
import os
from GetTradeDate import LocalTradeCalendar
from datetime import datetime, date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPathManager:
    """数据路径管理"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def get_file_path(self, file_name: str) -> Path:
        """获取文件路径"""
        return self.data_root / file_name

    def load_data(self, file_name: str) -> pd.DataFrame:
        """加载数据"""
        file_path = self.get_file_path(file_name)
        if file_path.exists():
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()


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
    def save_to_csv(df: pd.DataFrame, filepath: str):
        """保存DataFrame到CSV文件"""
        if df.empty:
            logger.warning(f"跳过空数据: {filepath}")
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"数据已保存至: {filepath}")


class DataProcessor:
    @staticmethod
    def process(today: str, data_root: str):
        """数据处理主流程"""
        logger.info("开始执行数据处理流程...")
        ltc = LocalTradeCalendar()
        start_date = ltc.get_previous_trade_date(today, 5)

        # 创建目录
        os.makedirs(data_root, exist_ok=True)

        # 获取并保存各类数据
        DataProcessor._save_zt_pool(today, data_root)
        DataProcessor._save_zt_pool_hist(today, data_root)
        DataProcessor._save_sector_list(today, data_root)
        DataProcessor._save_sector_history(today, start_date, data_root)
        DataProcessor._save_lhb_data(start_date, today, data_root)
        DataProcessor._save_index_data(data_root)
        DataProcessor._save_market_breadth(data_root)
        DataProcessor._save_market_volume(data_root)
        DataProcessor._save_zt_stock_data(today, start_date, data_root)
        DataProcessor._save_sector_components(today, data_root)  # 添加保存板块成分股数据

    @staticmethod
    def _save_zt_pool(today: str, data_root: str):
        """保存涨停池数据"""
        zt_df = DataFetcher.fetch_zt_pool(today)
        if not zt_df.empty:
            filepath = os.path.join(data_root, "zt_pool.csv")
            DataSaver.save_to_csv(zt_df, filepath)

    @staticmethod
    def _save_zt_pool_hist(today: str, data_root: str):
        """保存近N日涨停数据"""
        zt_df = DataFetcher.fetch_zt_pool_hist(end_date=today)
        zt_df = zt_df.reindex(columns=[
                            "序号", "日期", "代码", "名称", "涨跌幅", "最新价",
                            "成交额", "流通市值", "总市值", "换手率",
                            "封板资金", "首次封板时间", "最后封板时间", "炸板次数", "涨停统计", "连板数", "所属行业"
                        ])
        if not zt_df.empty:
            filepath = os.path.join(data_root, "zt_pool_hist.csv")
            DataSaver.save_to_csv(zt_df, filepath)

    @staticmethod
    def _save_sector_list(today: str, data_root: str):
        """保存板块列表数据"""
        sector_list_data = pd.DataFrame()

        for sector_type in ["industry", "concept"]:
            sector_list = DataFetcher.fetch_sector_list(sector_type)
            if not sector_list.empty:
                sector_list["板块类型"] = sector_type
                sector_list_data = pd.concat([sector_list_data, sector_list], ignore_index=True)

        if not sector_list_data.empty:
            filepath = os.path.join(data_root, "sector_list.csv")
            DataSaver.save_to_csv(sector_list_data, filepath)

    @staticmethod
    def _save_sector_history(today: str, start_date: str, data_root: str):
        """保存板块历史数据"""
        sector_hist_data = pd.DataFrame()

        for sector_type in ["industry", "concept"]:
            sector_list = DataFetcher.fetch_sector_list(sector_type)
            if not sector_list.empty:
                for _, row in sector_list.iterrows():
                    hist_df = DataFetcher.fetch_sector_history(
                        sector=row["板块名称"],
                        sector_type=sector_type,
                        start_date=start_date,
                        end_date=today
                    )
                    if not hist_df.empty:
                        hist_df["板块代码"] = row["板块代码"]
                        hist_df["板块名称"] = row["板块名称"]
                        hist_df["板块类型"] = sector_type
                        sector_hist_data = pd.concat([sector_hist_data, hist_df], ignore_index=True)

        if not sector_hist_data.empty:
            filepath = os.path.join(data_root, "sector_hist.csv")
            DataSaver.save_to_csv(sector_hist_data, filepath)

    @staticmethod
    def _save_lhb_data(start_date: str, today: str, data_root: str):
        """保存龙虎榜数据"""
        # 主表
        lhb_main = DataFetcher.fetch_lhb_detail(start_date, today)
        if lhb_main.empty:
            logger.warning("龙虎榜主表数据为空")
            return

        lhb_main.sort_values(by=["股票代码", "股票名称"], ascending=True, inplace=True)
        filepath = os.path.join(data_root, "lhb_main.csv")
        DataSaver.save_to_csv(lhb_main, filepath)

        # 明细表
        if not lhb_main.empty:
            lhb_details = []
            for _, row in lhb_main.iterrows():
                # 确保日期是字符串格式
                date_str = row["数据日期"].strftime("%Y%m%d") if isinstance(row["数据日期"], (date, datetime)) else row["数据日期"]
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
                filepath = os.path.join(data_root, "lhb_detail.csv")
                DataSaver.save_to_csv(combined_df, filepath)

    @staticmethod
    def _save_index_data(data_root: str):
        """保存指数数据"""
        index_codes = ["sh000001", "sz399001", "sz399006", "sh000016"]
        index_data = pd.DataFrame()

        for code in index_codes:
            index_df = DataFetcher.fetch_index_data(code)
            if not index_df.empty:
                index_df["指数代码"] = code
                index_data = pd.concat([index_data, index_df], ignore_index=True)

        if not index_data.empty:
            filepath = os.path.join(data_root, "index_data.csv")
            DataSaver.save_to_csv(index_data, filepath)

    @staticmethod
    def _save_market_breadth(data_root: str):
        """保存市场广度数据"""
        breadth_df = DataFetcher.fetch_market_breadth()
        if not breadth_df.empty:
            filepath = os.path.join(data_root, "market_breadth.csv")
            DataSaver.save_to_csv(breadth_df, filepath)

    @staticmethod
    def _save_market_volume(data_root: str):
        """保存市场成交量数据"""
        market_volume_data = pd.DataFrame()

        for market in ["sh", "sz", "bj"]:
            volume_df = DataFetcher.fetch_market_volume(market)
            if not volume_df.empty:
                volume_df["市场类型"] = market
                market_volume_data = pd.concat([market_volume_data, volume_df], ignore_index=True)

        if not market_volume_data.empty:
            filepath = os.path.join(data_root, "market_volume.csv")
            DataSaver.save_to_csv(market_volume_data, filepath)

    @staticmethod
    def _save_zt_stock_data(today: str, start_date: str, data_root: str):
        """保存当日涨停股票的历史数据"""
        # 获取当日涨停股票列表
        zt_df = DataFetcher.fetch_zt_pool_hist(end_date=today)
        if zt_df.empty:
            logger.warning("当日涨停股票数据为空")
            return

        # 提取涨停股票代码列表
        zt_codes = zt_df['代码'].unique().tolist()

        stock_hist_data = pd.DataFrame()
        stock_info_data = pd.DataFrame()

        # 基本面信息
        info_df = DataFetcher.fetch_stock_info()
        if not info_df.empty:
            stock_info_data = pd.concat([stock_info_data, info_df], ignore_index=True)
        if not stock_info_data.empty:
            filepath = os.path.join(data_root, "stock_info.csv")
            DataSaver.save_to_csv(stock_info_data, filepath)

        for code in zt_codes:
            # 历史K线
            hist_df = DataFetcher.fetch_stock_hist(code, start_date, today)
            if not hist_df.empty:
                hist_df["股票代码"] = code
                stock_hist_data = pd.concat([stock_hist_data, hist_df], ignore_index=True)

        if not stock_hist_data.empty:
            filepath = os.path.join(data_root, "zt_stock_hist.csv")
            DataSaver.save_to_csv(stock_hist_data, filepath)

    @staticmethod
    def _save_sector_components(today: str, data_root: str):
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
            filepath = os.path.join(data_root, "sector_components.csv")
            DataSaver.save_to_csv(sector_components_data, filepath)


def main():
    ltc = LocalTradeCalendar()
    today = ltc.get_recent_trade_date()
    data_root = os.path.join("data/csv", today)
    DataProcessor.process(today, data_root)
    logger.info(f"\n=== 数据存储完成 ===")
    logger.info(f"文件存储路径: {data_root}")


if __name__ == "__main__":
    main()
