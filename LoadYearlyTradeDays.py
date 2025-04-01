import pandas_market_calendars as mcal
import pandas as pd


def save_trading_days_to_csv():
    try:
        # 获取上交所交易日历
        sh_calendar = mcal.get_calendar('SSE')

        # 定义日期范围（2025年全年）
        start_date = '2025-01-01'
        end_date = '2025-12-31'

        # 获取交易日历
        schedule = sh_calendar.schedule(start_date=start_date, end_date=end_date)

        # 处理交易日数据（转换为标准日期格式）
        trading_days = pd.DataFrame({
            'trading_date': schedule.index.strftime('%Y%m%d'),
            'is_open': True  # 标记为交易日
        })

        # 保存路径（当前目录下的data文件夹）
        file_path = './data/sse_trading_days_2025.csv'

        # 保存为CSV（带BOM避免中文乱码）
        trading_days.to_csv(file_path, index=False, encoding='utf-8-sig')

        print(f"✅ 成功保存 {len(trading_days)} 个交易日到：{file_path}")
        return True

    except Exception as e:
        print(f"❌ 保存失败：{str(e)}")
        return False


if __name__ == "__main__":
    save_trading_days_to_csv()