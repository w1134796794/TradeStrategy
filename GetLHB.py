import akshare as ak
import pandas as pd
from pathlib import Path
from typing import List
from datetime import datetime, date
from GetTradeDate import LocalTradeCalendar


# --------------------------
# 配置类
# --------------------------
class LHBConfig:
    """龙虎榜数据配置"""
    BASE_PATH = Path("data/lhb")  # 基础存储路径
    SUB_DIRS = {  # 子目录结构
        'daily': 'daily',  # 每日基础数据
        'detail': 'detail',  # 营业部明细数据
        'stat': 'stat',  # 统计周期数据
        'result': 'result'  # 最终结果数据
    }
    MAX_RETRY = 3  # 最大重试次数
    STAT_PERIODS = ['近3日', '近5日', '近10日', '近一月']  # 统计周期配置


# --------------------------
# 缓存目录管理
# --------------------------
class LHBDataManager:
    """龙虎榜数据目录管理"""

    def __init__(self):
        self.base_path = LHBConfig.BASE_PATH
        self.sub_dirs = {
            k: self.base_path / v for k, v in LHBConfig.SUB_DIRS.items()
        }
        self._create_dirs()

    def _create_dirs(self):
        """创建必要的目录结构"""
        for dir_path in self.sub_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_file_path(self, data_type: str, date: str, period: str = None) -> Path:
        """获取数据存储路径"""
        if data_type == 'stat':
            return self.sub_dirs['stat'] / f"{period}_{date}.parquet"
        return self.sub_dirs[data_type] / f"{date}.parquet"

    def save_data(self, df: pd.DataFrame, data_type: str, date: str, period: str = None):
        """保存数据到对应目录"""
        file_path = self.get_file_path(data_type, date, period)
        df.to_parquet(file_path)
        print(f"数据已保存至: {file_path}")

    def load_data(self, data_type: str, date: str, period: str = None) -> pd.DataFrame:
        """从本地加载数据"""
        file_path = self.get_file_path(data_type, date, period)
        if file_path.exists():
            return pd.read_parquet(file_path)
        return pd.DataFrame()


# --------------------------
# 核心分析类
# --------------------------
class LHBAnalyzer:
    def __init__(self):
        # 增加调试日志
        self.debug_mode = True
        self.max_retry = LHBConfig.MAX_RETRY
        self.data_mgr = LHBDataManager()
        self.calendar = LocalTradeCalendar()
        self.statistics_cache = {}

    def _log(self, message):
        """添加调试日志"""
        if self.debug_mode:
            print(f"[DEBUG] {datetime.now().strftime('%H:%M:%S')} | {message}")

    def _fetch_api_data(self, func, **kwargs):
        """增强版API请求方法"""
        for i in range(LHBConfig.MAX_RETRY):
            try:
                result = func(**kwargs)
                self._log(f"API {func.__name__} 返回类型: {type(result)}")

                # 深度检查返回数据结构
                if isinstance(result, pd.DataFrame):
                    if result.empty:
                        self._log("返回空DataFrame")
                        return pd.DataFrame()
                    self._log(f"数据列: {result.columns.tolist()}")
                    return result
                elif result is None:
                    self._log("API返回None")
                    return pd.DataFrame()
                else:
                    self._log(f"意外返回类型: {type(result)}")
                    return pd.DataFrame()
            except KeyError as e:
                print(f"数据结构异常: {str(e)}")
                return pd.DataFrame()
            except Exception as e:
                print(f"API请求失败[{i + 1}/{LHBConfig.MAX_RETRY}]: {str(e)}")
        return pd.DataFrame()

    def _get_basic_data(self, date: str) -> pd.DataFrame:
        """增强基础数据获取"""
        df = self.data_mgr.load_data('daily', date)
        if not df.empty:
            self._log(f"从缓存加载基础数据 {date}")
            return df

        new_df = self._fetch_api_data(ak.stock_lhb_detail_em, start_date=date, end_date=date)
        if new_df.empty:
            return pd.DataFrame()

        # 动态处理列名
        required_columns = {
            '代码': 'code_column',
            '名称': 'name_column',
            '上榜日': 'date_column'
        }

        # 列名映射表（根据akshare实际返回修改）
        column_mapping = {
            'code_column': '代码',
            'name_column': '名称',
            'date_column': '上榜日',
            'interpretation_column': '解读',
            'change_rate_column': '涨跌幅',
            'reason_column': '上榜原因'
        }

        # 自动适配列名
        actual_columns = new_df.columns.tolist()
        for standard_name in required_columns:
            if column_mapping.get(required_columns[standard_name]) in actual_columns:
                new_df.rename(columns={column_mapping[required_columns[standard_name]]: standard_name}, inplace=True)

        # 强制类型转换
        try:
            new_df['代码'] = new_df['代码'].astype(str).str.zfill(6)
        except KeyError:
            self._log("代码列不存在")
            return pd.DataFrame()

        # 保存前验证必要列
        essential_cols = ['代码', '名称', '上榜日']
        if not all(col in new_df.columns for col in essential_cols):
            self._log(f"缺失必要列，实际列: {new_df.columns}")
            return pd.DataFrame()

        self.data_mgr.save_data(new_df[essential_cols + ['解读', '涨跌幅', '上榜原因']], 'daily', date)
        return new_df

    def _get_trade_details(self, code: str, date: str) -> pd.DataFrame:
        """增强营业部数据获取"""

        def safe_get(d, key, default=0):
            """安全获取嵌套值"""
            if isinstance(d, dict):
                return d.get(key, default)
            return default

        # 获取原始数据
        raw_df = self.data_mgr.load_data('detail', date)
        if not raw_df.empty:
            return raw_df

        detail_dfs = []
        for flag in ["买入", "卖出"]:
            df = self._fetch_api_data(
                ak.stock_lhb_stock_detail_em,
                symbol=code,
                date=date,
                flag=flag
            )
            if not df.empty:
                # 动态处理列名
                formatted_df = df.rename(columns={
                    'trade_department': '营业部名称',
                    'buy_amount': '买入金额',
                    'buy_ratio': '买入占比',
                    'sell_amount': '卖出金额',
                    'sell_ratio': '卖出占比'
                }, errors='ignore')
                formatted_df['代码'] = code
                detail_dfs.append(formatted_df)

        if not detail_dfs:
            return pd.DataFrame()

        combined = pd.concat(detail_dfs)
        # 填充可能缺失的列
        for col in ['买入金额', '买入占比', '卖出金额', '卖出占比']:
            if col not in combined.columns:
                combined[col] = 0.0

        self.data_mgr.save_data(combined, 'detail', date)
        return combined

    # --------------------------
    # 分析计算方法
    # --------------------------
    def _enrich_statistics(self, main_df: pd.DataFrame, period: str) -> pd.DataFrame:
        stat_df = self._fetch_api_data(ak.stock_lhb_stock_statistic_em, symbol=period)
        if stat_df.empty:
            return main_df

        stat_df['代码'] = stat_df['代码'].astype(str).str.zfill(6)
        stat_df['最近上榜日'] = pd.to_datetime(stat_df['最近上榜日'], format='%Y%m%d', errors='coerce')

        merged = main_df.merge(
            stat_df.groupby('代码').agg(
                累计上榜次数=('代码', 'count'),
                最近上榜日=('最近上榜日', 'max')
            ).reset_index(),
            on='代码',
            how='left'
        )

        merged['上榜日'] = pd.to_datetime(merged['上榜日'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        merged['最近上榜日'] = merged['最近上榜日'].dt.strftime('%Y-%m-%d')

        return merged.fillna({'累计上榜次数': 0})

    def _process_org_data(self, main_df: pd.DataFrame) -> pd.DataFrame:
        """安全处理嵌套数据"""
        expanded = []
        for _, row in main_df.iterrows():
            # 安全获取基础字段
            base_info = {
                '代码': row.get('代码', '未知代码'),
                '名称': row.get('名称', '未知名称'),
                '上榜日': row.get('上榜日', '未知日期'),
                '解读': row.get('解读', ''),
                '涨跌幅': row.get('涨跌幅', 0),
                '上榜原因': row.get('上榜原因', '')
            }

            # 处理营业部数据
            org_data = row.get('营业部数据', [])
            if not isinstance(org_data, list):
                continue

            for detail in org_data:
                if not isinstance(detail, dict):
                    continue

                record = {
                    **base_info,
                    '营业部名称': detail.get('营业部名称', '未知席位'),
                    '买入金额': float(detail.get('买入金额', 0)),
                    '买入占比': float(detail.get('买入占比', 0)),
                    '卖出金额': float(detail.get('卖出金额', 0)),
                    '卖出占比': float(detail.get('卖出占比', 0))
                }
                expanded.append(record)

        return pd.DataFrame(expanded)

    # --------------------------
    # 主流程方法
    # --------------------------
    def get_enhanced_data(self, dates: List[str], period: str = "近一月") -> pd.DataFrame:
        valid_dates = []
        for d in dates:
            try:
                parsed_date = datetime.strptime(d, "%Y%m%d").date()
                if parsed_date > date.today():
                    print(f"跳过未来日期: {d}")
                    continue
                valid_dates.append(d)
            except ValueError:
                print(f"无效日期: {d}")

        if not valid_dates:
            return pd.DataFrame()

        base_data = []
        for date_str in valid_dates:
            daily_df = self._get_basic_data(date_str)
            for _, row in daily_df.iterrows():
                detail_df = self._get_trade_details(row['代码'], date_str)
                if not detail_df.empty:
                    merged_row = row.to_dict()
                    merged_row['营业部数据'] = detail_df.to_dict('records')
                    base_data.append(merged_row)

        if not base_data:
            return pd.DataFrame()

        processed_df = self._process_org_data(pd.DataFrame(base_data))
        final_df = self._enrich_statistics(processed_df, period)

        num_cols = ['涨跌幅', '买入金额', '买入占比', '卖出金额', '卖出占比']
        final_df[num_cols] = final_df[num_cols].apply(pd.to_numeric, errors='coerce')

        if '净买入额' not in final_df.columns:
            final_df['净买入额'] = final_df['买入金额'] - final_df['卖出金额']

        date_cols = ['上榜日', '最近上榜日']
        for col in date_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col], errors='coerce').dt.strftime('%Y-%m-%d')

        self.data_mgr.save_data(final_df, 'result', valid_dates[-1])
        return final_df.dropna(subset=['营业部名称'])


if __name__ == "__main__":
    analyzer = LHBAnalyzer()
    analyzer.debug_mode = True  # 开启调试模式

    # 测试日期（使用真实存在的日期）
    test_dates = [datetime.now().strftime("%Y%m%d")]  # 当前日期

    result = analyzer.get_enhanced_data(test_dates)

    if not result.empty:
        print("处理成功，结果样例:")
        print(result.head())
    else:
        print("无有效数据，请检查：")
        print("1. 网络连接是否正常")
        print("2. 测试日期是否为交易日")
        print("3. 目标股票是否实际上榜")