import time
from pathlib import Path
import pickle
from GenerateTradePlan import TradePlanGenerator


def main():
    cache_file = Path("last_plan.pkl")
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime < 300):
        try:
            with open(cache_file, "rb") as f:
                plan = pickle.load(f)
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
            generator = TradePlanGenerator()
            plan = generator.generate_daily_plan()
            with open(cache_file, "wb") as f:
                pickle.dump(plan, f)
    else:
        generator = TradePlanGenerator()
        plan = generator.generate_daily_plan()
        with open(cache_file, "wb") as f:
            pickle.dump(plan, f)
    generator = TradePlanGenerator()
    plan = generator.generate_daily_plan()

    print(f"== 交易计划 {plan['plan_date']} ==")
    print(f"市场状态: {plan['market_status']['level']} ({plan['market_status']['score']}分)")
    print(f"风险提示: {plan['risk_assessment']}")

    print("\n推荐标的:")
    for stock in plan['candidates']:
        print(f"{stock['name']}({stock['code']}) | 评分: {stock['score']} | 建议仓位: {stock['position']}%")
        print(f"　　入场价: ¥{stock['entry_price']:.2f} | 止损价: ¥{stock['stop_loss']:.2f}")
        print(f"　　推荐理由: {stock['reasons']}")


if __name__ == "__main__":
    generator = TradePlanGenerator()
    plan = generator.generate_daily_plan()

    print(f"== 交易计划 {plan['plan_date']} ==")
    print(f"市场状态: {plan['market_status']['level']} ({plan['market_status']['score']}分)")
    print(f"风险提示: {plan['risk_assessment']}")

    print("\n推荐标的:")
    for stock in plan['candidates']:
        print(f"{stock['name']}({stock['code']}) | 评分: {stock['score']} | 建议仓位: {stock['position']}%")
        print(f"　　入场价: ¥{stock['entry_price']:.2f} | 止损价: ¥{stock['stop_loss']:.2f}")
        print(f"　　推荐理由: {stock['reasons']}")

