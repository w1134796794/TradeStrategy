import akshare as ak
import pandas as pd

pd.set_option("display.max_column", None)


stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20250310", end_date='20250327')
print(stock_zh_a_hist_df)