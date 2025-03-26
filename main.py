from GenerateTradePlan import TradePlanGenerator

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