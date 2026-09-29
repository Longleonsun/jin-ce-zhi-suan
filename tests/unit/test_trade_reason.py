"""成交记录保存原因，报告能区分策略信号与系统操作（回测结束平仓、总回撤强平、止损止盈）。"""

import asyncio

import numpy as np
import pandas as pd

from src.core.backtest_cabinet import BacktestCabinet
from src.core.shangshu_sheng import ShangshuSheng
from src.core.zhongshu_sheng import ZhongshuSheng
from src.ministries.bing_bu_war import BingBuWar
from src.ministries.hu_bu_revenue import HuBuRevenue
from src.ministries.xing_bu_justice import XingBuJustice
from src.strategies.implemented_strategies import Strategy00, Strategy13

DEFAULTS_13 = {"entry_period": 25, "exit_period": 12, "vol_window": 20, "target_vol": 0.40,
               "max_position_pct": 1.0, "rebalance_threshold": 0.1}


class _Provider:
    def __init__(self, df):
        self.df, self.last_error = df, ""

    def check_connectivity(self, code):
        return True, "ok"

    def fetch_kline_data(self, code, start_time, end_time, interval="D"):
        return self.df.copy()


def _bars(closes, code="510300.SH"):
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"code": code, "dt": pd.bdate_range("2024-01-01", periods=len(c)), "open": c,
                         "high": c * 1.01, "low": c * 0.99, "close": c, "vol": 1e6, "amount": 1e8})


def _run(monkeypatch, df, strategies, max_drawdown_pct=0.0):
    events = []

    async def _collect(event_type, data):
        if event_type in ("backtest_trade", "backtest_strategy_report"):
            events.append((event_type, data))

    cab = BacktestCabinet(df["code"].iloc[0], strategy_ids=["00"], provider_override=_Provider(df),
                          provider_source_override="unit_stub", event_callback=_collect)
    for s in strategies:
        if isinstance(s, Strategy13):
            monkeypatch.setattr(s, "_cfg", lambda key, default: DEFAULTS_13.get(key, default))
    cab.strategies = list(strategies)
    cab.secretariat = ZhongshuSheng(cab.strategies)
    cab.strategy_initial_capital = cab.initial_capital / len(strategies)
    cab.strategy_revenues = {s.id: HuBuRevenue(cab.strategy_initial_capital) for s in cab.strategies}
    base_get = cab.config.get
    monkeypatch.setattr(cab.config, "get", lambda key, default=None: max_drawdown_pct
                        if key == "risk_control.max_drawdown_pct" else base_get(key, default))

    async def _no_persist(*a, **k):
        return None

    monkeypatch.setattr(cab, "_persist_backtest_cache_to_db", _no_persist)
    monkeypatch.setattr(BacktestCabinet, "_tf_cache", {})
    asyncio.run(cab.run(start_date=df["dt"].min(), end_date=df["dt"].max()))
    trades = [d for t, d in events if t == "backtest_trade"]
    reports = {d["strategy_id"]: d for t, d in events if t == "backtest_strategy_report"}
    return cab, trades, reports


def test_force_close_at_end_is_labeled(monkeypatch):
    cab, trades, reports = _run(monkeypatch, _bars([10 + 0.01 * i for i in range(60)]), [Strategy00()])
    last = cab.strategy_revenues["00"].transactions[-1]
    assert last["direction"] == "SELL" and last["reason"] == "FORCE_CLOSE_END"
    assert trades[-1]["reason"] == "FORCE_CLOSE_END"
    report = reports["00"]
    assert report["force_close_count"] == 1
    assert report["trade_details"][-1]["reason_label"] == "回测结束平仓"
    assert any("最后一笔" in n and "强制平仓" in n for n in report["report_notes"])


def test_drawdown_liquidation_and_strategy_exit_are_distinguished(monkeypatch):
    # 上涨 → 单日跳水 10% 触发 6% 总回撤强平 → 再涨回去重新入场 → 缓跌触发唐奇安离场
    closes = [10 * 1.004 ** i for i in range(120)]
    closes += [closes[-1] * 0.9] * 5
    closes += [closes[-1] * 1.01 ** (i + 1) for i in range(40)]
    closes += [closes[-1] * 0.99 ** (i + 1) for i in range(20)]
    cab, trades, reports = _run(monkeypatch, _bars(closes), [Strategy13()], max_drawdown_pct=0.06)
    reasons = [t["reason"] for t in cab.strategy_revenues["13"].transactions if t["direction"] == "SELL"]
    assert "DRAWDOWN_LIMIT" in reasons
    assert "Donchian Exit" in reasons
    labels = {d["reason"]: d["reason_label"] for d in reports["13"]["trade_details"]}
    assert labels["DRAWDOWN_LIMIT"] == "总回撤强平"
    assert labels["Donchian Exit"] == "策略信号"
    assert any("总回撤强平" in n for n in reports["13"]["report_notes"])


def test_stop_loss_order_records_reason():
    hu_bu = HuBuRevenue(1_000_000)
    shangshu = ShangshuSheng(hu_bu, BingBuWar(), XingBuJustice())
    shangshu.positions = {"T1": {"AAPL.US": {"qty": 100, "avg_price": 100.0, "direction": "BUY",
                                             "stop_loss": 95.0, "take_profit": None,
                                             "lots": [{"qty": 100, "buy_day": "2024-01-02", "unit_cost": 100.0}]}}}
    bar = {"code": "AAPL.US", "dt": pd.Timestamp("2024-01-03"), "open": 99.0, "high": 99.5,
           "low": 94.0, "close": 94.5, "vol": 1e6}
    orders = shangshu.check_stops(bar)
    assert orders[0]["reason"] == "STOP_LOSS"
    assert shangshu.execute_order("T1", orders[0], bar, hu_bu_account=hu_bu)
    assert hu_bu.transactions[-1]["reason"] == "STOP_LOSS"
