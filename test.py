import akshare as ak
import pandas as pd

pd.set_option("display.max_column", None)

df = ak.stock_zh_a_spot_em()
print(df)