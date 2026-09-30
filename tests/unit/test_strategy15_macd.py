"""15 号策略：MACD(3,6,5) 金叉买入、死叉卖出。"""

import numpy as np
import pandas as pd

from src.strategies.implemented_strategies import Strategy15
from src.strategies.strategy_factory import create_strategies
from src.strategies.strategy_manager_repo import _BUILTIN_STRATEGY_CLASSES


def _new(monkeypatch, cash=1_000_000, **cfg):
    s = Strategy15()
    params = {"fast_period": 3, "slow_period": 6, "signal_period": 5,
              "order_qty_mode": "cash_pct", "order_cash_pct": 100, **cfg}
    monkeypatch.setattr(s, "_cfg", lambda key, default: params.get(key, default))
    s.set_backtest_context(current_cash=cash, total_value=cash)
    return s


def _bar(code, i, close):
    return {"code": code, "dt": pd.Timestamp("2024-01-01") + pd.offsets.BDay(i), "close": close}


def _feed(s, code, closes):
    """逐根推送；按信号数量立即更新持仓，模拟成交。"""
    sigs = []
    for i, c in enumerate(closes):
        s.last_price = c
        sig = s.on_bar(_bar(code, i, c))
        if sig:
            qty = int(s.positions.get(code, 0))
            s.update_position(code, qty + sig["qty"] if sig["direction"] == "BUY" else qty - sig["qty"])
        sigs.append(sig)
    return sigs


def _reference(closes, fast=3, slow=6, signal=9, warmup=None):
    c = pd.Series(closes, dtype=float)
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    d = (dif - dif.ewm(span=signal, adjust=False).mean()).to_numpy()
    warmup = 3 * (slow + signal) if warmup is None else warmup
    hold, out = False, []
    for i in range(len(d)):
        if i >= warmup + 1:
            if hold and d[i - 1] >= 0 > d[i]:
                hold = False
            elif not hold and d[i - 1] <= 0 < d[i]:
                hold = True
        out.append(hold)
    return out


def test_positions_match_reference_state_machine(monkeypatch):
    rng = np.random.default_rng(3)
    closes = list(10 * np.cumprod(1 + rng.normal(0.0005, 0.02, 600)))
    s = _new(monkeypatch)
    _feed(s, "510300.SH", closes)
    ref = _reference(closes, signal=5)
    held = []
    s2 = _new(monkeypatch)
    for i, c in enumerate(closes):
        s2.last_price = c
        sig = s2.on_bar(_bar("510300.SH", i, c))
        if sig:
            qty = int(s2.positions.get("510300.SH", 0))
            s2.update_position("510300.SH", qty + sig["qty"] if sig["direction"] == "BUY" else qty - sig["qty"])
        held.append(s2.positions.get("510300.SH", 0) > 0)
    assert held == ref
    assert sum(1 for a, b in zip(ref[1:], ref[:-1]) if a and not b) > 20  # 有足够多的买入样本


def test_golden_cross_buys_and_dead_cross_sells_all(monkeypatch):
    s = _new(monkeypatch)
    closes = [10.0 - 0.05 * i for i in range(40)] + [8.2, 8.6, 9.0, 9.4] + [9.0, 8.5, 8.0]
    sigs = _feed(s, "510300.SH", closes)
    buys = [(i, x) for i, x in enumerate(sigs) if x and x["direction"] == "BUY"]
    sells = [(i, x) for i, x in enumerate(sigs) if x and x["direction"] == "SELL"]
    assert buys and sells and buys[0][0] < sells[0][0]
    i, buy = buys[0]
    assert buy["qty"] == int(1_000_000 // closes[i] // 100 * 100)
    assert sells[0][1]["qty"] == buy["qty"] and sells[0][1]["reason"] == "MACD Dead Cross"


def test_no_signal_during_warmup(monkeypatch):
    s = _new(monkeypatch)
    zigzag = [10 + (0.5 if i % 4 < 2 else -0.5) for i in range(3 * (6 + 5) + 1)]
    assert all(x is None for x in _feed(s, "510300.SH", zigzag))


def test_repeated_bar_push_is_idempotent(monkeypatch):
    closes = [10.0 - 0.05 * i for i in range(40)] + [8.2, 8.6, 9.0]
    k = next(i for i, x in enumerate(_feed(_new(monkeypatch), "510300.SH", closes)) if x)  # 首次金叉所在K线
    s = _new(monkeypatch)
    _feed(s, "510300.SH", closes[:k])
    s.last_price = closes[k]
    bar = _bar("510300.SH", k, closes[k])
    first = s.on_bar(bar)
    assert first is not None and first["direction"] == "BUY"
    assert s.on_bar(dict(bar)) == first


def test_periods_are_configurable(monkeypatch):
    rng = np.random.default_rng(5)
    closes = list(10 * np.cumprod(1 + rng.normal(0.0005, 0.02, 400)))
    s = _new(monkeypatch, fast_period=12, slow_period=26, signal_period=9)
    _feed(s, "510300.SH", closes)
    s2 = _new(monkeypatch, fast_period=12, slow_period=26, signal_period=9)
    held = []
    for i, c in enumerate(closes):
        s2.last_price = c
        sig = s2.on_bar(_bar("510300.SH", i, c))
        if sig:
            qty = int(s2.positions.get("510300.SH", 0))
            s2.update_position("510300.SH", qty + sig["qty"] if sig["direction"] == "BUY" else qty - sig["qty"])
        held.append(s2.positions.get("510300.SH", 0) > 0)
    assert held == _reference(closes, fast=12, slow=26, signal=9)


def test_registered_as_builtin():
    assert _BUILTIN_STRATEGY_CLASSES["15"] is Strategy15
    assert "15" in {s.id for s in create_strategies(apply_active_filter=False)}
