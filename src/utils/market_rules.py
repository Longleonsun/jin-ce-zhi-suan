# src/utils/market_rules.py
"""标的所属市场规则：下单单位（A股整百股，美股1股）与交易时段。"""

_US_SUFFIXES = (".US", ".O", ".N", ".NASDAQ", ".NYSE")


def is_us_symbol(code):
    c = str(code or "").strip().upper()
    if c.endswith(_US_SUFFIXES):
        return True
    return bool(c) and c.replace("-", "").replace(".", "").isalpha()


def lot_size_for_code(code):
    return 1 if is_us_symbol(code) else 100


def _hm(hhmm):
    """930 -> 570（当日分钟数）。"""
    return (int(hhmm) // 100) * 60 + int(hhmm) % 100


# 时间均为交易所当地时间，分钟K线按结束时间标记（9:31 表示 9:30-9:31）。
# daily_trigger：日线策略触发窗口 [daily_trigger, close)，窗口内第一根K线触发、每日一次。
# A股 14:57 起为收盘集合竞价，Yahoo 等数据源不提供 14:58 之后的分钟线，故从 14:57 开始；
# 美股从 15:58 开始，留出一分钟余量以防个别分钟线缺失或无法被下一分钟确认。
_MARKETS = {
    "CN": {
        "tz": "Asia/Shanghai",
        "sessions": ((_hm(930), _hm(1130)), (_hm(1300), _hm(1500))),
        "close": _hm(1500),
        "daily_trigger": _hm(1457),
        "summary": _hm(1505),
        "auto_stop": _hm(1530),
        "t_plus_one": True,
        "cost_config_key": "trading_cost",
        "cost_defaults": {"commission_rate": 0.00025, "min_commission": 5.0, "stamp_duty": 0.001, "transfer_fee": 0.0},
    },
    "US": {
        "tz": "America/New_York",
        "sessions": ((_hm(930), _hm(1600)),),
        "close": _hm(1600),
        "daily_trigger": _hm(1558),
        "summary": _hm(1605),
        "auto_stop": _hm(1630),
        "t_plus_one": False,
        # 美股无印花税；卖出收 SEC 费（约十万分之 2.78），佣金按券商设置
        "cost_config_key": "trading_cost_us",
        "cost_defaults": {"commission_rate": 0.0, "min_commission": 0.0, "stamp_duty": 0.0000278, "transfer_fee": 0.0},
    },
}


def market_of(code):
    return "US" if is_us_symbol(code) else "CN"


def market_profile(code):
    return _MARKETS[market_of(code)]


def is_t_plus_one(code):
    return market_profile(code)["t_plus_one"]
