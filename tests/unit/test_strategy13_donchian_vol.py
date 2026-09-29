"""13 号策略：唐奇安趋势择时 + 波动率目标定仓；以及与 00 号长期持有同区间对比回测。"""

import asyncio

import numpy as np
import pandas as pd
import pytest

from src.core.backtest_cabinet import BacktestCabinet
from src.core.zhongshu_sheng import ZhongshuSheng
from src.ministries.hu_bu_revenue import HuBuRevenue
from src.strategies.implemented_strategies import Strategy00, Strategy13

DEFAULTS = {"entry_period": 25, "exit_period": 12, "vol_window": 20, "target_vol": 0.40,
            "max_position_pct": 1.0, "rebalance_threshold": 0.1}


def _new(monkeypatch, total=1_000_000, **cfg):
    s = Strategy13()
    params = {**DEFAULTS, **cfg}
    monkeypatch.setattr(s, "_cfg", lambda key, default: params.get(key, default))
    s.set_backtest_context(current_cash=total, total_value=total)
    return s


def _bar(code, i, close):
    dt = pd.Timestamp("2024-01-01") + pd.offsets.BDay(i)
    return {"code": code, "dt": dt, "open": close, "high": close, "low": close, "close": close, "vol": 1e6}


def _feed(s, code, closes, fill=True):
    """逐根推送收盘价；fill=True 时按信号数量立即更新持仓，模拟成交。"""
    signals = []
    for i, c in enumerate(closes):
        sig = s.on_bar(_bar(code, i, c))
        signals.append(sig)
        if sig and fill:
            qty = int(s.positions.get(code, 0))
            s.update_position(code, qty + sig["qty"] if sig["direction"] == "BUY" else qty - sig["qty"])
    return signals


def _flat_then_breakout(n_flat=40, base=10.0):
    """小幅震荡（低波动）后突破。"""
    closes = [base * (1 + 0.002 * ((-1) ** i)) for i in range(n_flat)]
    return closes + [base * 1.03]


# ---------- 与参考实现（向量化，同 agucc V4 回测）逐根一致 ----------

def _reference(closes, entry=25, exit_=12, win=20, target=0.40, cap=1.0):
    c = pd.Series(closes, dtype=float)
    hi = c.rolling(entry).max().shift(1)
    lo = c.rolling(exit_).min().shift(1)
    hold, states = False, []
    for i in range(len(c)):
        if not np.isnan(hi.iloc[i]) and not np.isnan(lo.iloc[i]):
            if not hold and c.iloc[i] > hi.iloc[i]:
                hold = True
            elif hold and c.iloc[i] < lo.iloc[i]:
                hold = False
        states.append(hold)
    rv = c.pct_change().rolling(win).std(ddof=0) * np.sqrt(252)
    frac = (target / rv).clip(upper=cap)
    return states, frac


def test_state_and_position_match_reference(monkeypatch):
    rng = np.random.default_rng(7)
    rets = np.concatenate([rng.normal(0.002, 0.01, 120), rng.normal(-0.003, 0.035, 80), rng.normal(0.0015, 0.02, 120)])
    closes = list(10 * np.cumprod(1 + rets))
    s = _new(monkeypatch, total=1e12)  # 资金足够大，取整误差可忽略
    ref_hold, ref_frac = _reference(closes)
    held_frac = []
    for i, c in enumerate(closes):
        sig = s.on_bar(_bar("510300.SH", i, c))
        if sig:
            qty = int(s.positions.get("510300.SH", 0))
            s.update_position("510300.SH", qty + sig["qty"] if sig["direction"] == "BUY" else qty - sig["qty"])
        if i >= 25:
            assert s.hold_state.get("510300.SH", False) == ref_hold[i], f"bar {i}"
        held_frac.append(s.positions.get("510300.SH", 0) * c / 1e12)
    assert any(ref_hold) and not all(ref_hold[25:])
    # 持有期间实际仓位与参考目标仓位的偏离不超过调仓阈值
    for i in range(25, len(closes)):
        expect = ref_frac.iloc[i] if ref_hold[i] else 0.0
        assert abs(held_frac[i] - expect) < 0.1 + 1e-6, f"bar {i}: {held_frac[i]:.3f} vs {expect:.3f}"


# ---------- 入场、定仓、离场 ----------

def test_no_signal_before_enough_history(monkeypatch):
    s = _new(monkeypatch)
    assert all(sig is None for sig in _feed(s, "510300.SH", [10.0 + 0.01 * i for i in range(25)]))


def test_breakout_with_low_vol_buys_full_position_in_board_lots(monkeypatch):
    s = _new(monkeypatch)
    closes = _flat_then_breakout()
    sig = _feed(s, "510300.SH", closes)[-1]
    assert sig["direction"] == "BUY"
    assert sig["qty"] == int(1_000_000 // closes[-1] // 100 * 100)
    assert sig["stop_loss"] is None  # 不设止损价，避免触发门下省单笔止损幅度限制


def test_high_vol_scales_position_down(monkeypatch):
    s = _new(monkeypatch)
    closes = [10.0]
    for i in range(60):  # 上涨趋势中大幅震荡，年化波动远高于 40%
        closes.append(closes[-1] * (1.06 if i % 2 == 0 else 0.96))
    sigs = _feed(s, "510300.SH", closes)
    first_buy = next(i for i, x in enumerate(sigs) if x and x["direction"] == "BUY")
    _, frac = _reference(closes[:first_buy + 1])
    assert 0.2 < frac.iloc[-1] < 0.8
    expect = int(1_000_000 * frac.iloc[-1] // closes[first_buy] // 100 * 100)
    assert sigs[first_buy]["qty"] == expect


def test_break_below_exit_channel_sells_all(monkeypatch):
    s = _new(monkeypatch)
    closes = _flat_then_breakout() + [10.35, 10.4, 9.5]
    sigs = _feed(s, "510300.SH", closes)
    held = sigs[40]["qty"]
    assert sigs[-1]["direction"] == "SELL" and sigs[-1]["qty"] == held
    assert s.positions["510300.SH"] == 0


def test_small_deviation_does_not_rebalance(monkeypatch):
    s = _new(monkeypatch)
    sigs = _feed(s, "510300.SH", _flat_then_breakout() + [10.31, 10.33, 10.32])
    assert sigs[40]["direction"] == "BUY"
    assert sigs[41:] == [None, None, None]


def test_repeated_bar_push_is_idempotent(monkeypatch):
    s = _new(monkeypatch)
    closes = _flat_then_breakout()
    _feed(s, "510300.SH", closes[:-1])
    bar = _bar("510300.SH", len(closes) - 1, closes[-1])
    first = s.on_bar(bar)
    again = s.on_bar(dict(bar))
    assert first == again and first["direction"] == "BUY"
    assert s.hold_state["510300.SH"] is True


def test_us_symbol_uses_single_share_lot(monkeypatch):
    s = _new(monkeypatch)
    closes = [c * 33.3 for c in _flat_then_breakout()]
    sig = _feed(s, "AAPL", closes)[-1]
    assert sig["qty"] == int(1_000_000 // closes[-1])
    assert sig["qty"] % 100 != 0


# ---------- 与 00 号长期持有同区间对比回测 ----------

class _Provider:
    def __init__(self, df):
        self.df = df
        self.last_error = ""

    def check_connectivity(self, code):
        return True, "ok"

    def fetch_kline_data(self, code, start_time, end_time, interval="D"):
        return self.df.copy()


def _rise_then_crash(code):
    closes = [10 * 1.004 ** i for i in range(120)]
    closes += [closes[-1] * 0.99 ** (i + 1) for i in range(60)]
    dts = pd.bdate_range("2024-01-01", periods=len(closes))
    c = np.asarray(closes)
    # 留出日内振幅，避免四价相同被兵部识别为一字涨跌停而拒绝成交
    return pd.DataFrame({"code": code, "dt": dts, "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "vol": 1e6, "amount": 1e8})


def _rise_then_gap_down(code):
    """上涨后单日跳水 10%，随后横盘：用于触发 6% 总回撤强平。"""
    closes = [10 * 1.004 ** i for i in range(120)]
    closes += [closes[-1] * 0.90 * (1 + 0.001 * ((-1) ** i)) for i in range(30)]
    dts = pd.bdate_range("2024-01-01", periods=len(closes))
    c = np.asarray(closes)
    return pd.DataFrame({"code": code, "dt": dts, "open": c, "high": c * 1.01, "low": c * 0.99,
                         "close": c, "vol": 1e6, "amount": 1e8})


def _run(monkeypatch, df, strategies, max_drawdown_pct):
    """同一区间同跑多个策略；返回回测对象与成交事件。"""
    trades = []

    async def _collect(event_type, data):
        if event_type == "backtest_trade":
            trades.append(data)

    cab = BacktestCabinet(df["code"].iloc[0], strategy_ids=["00"], provider_override=_Provider(df),
                          provider_source_override="unit_stub", event_callback=_collect)
    for s in strategies:
        if isinstance(s, Strategy13):
            monkeypatch.setattr(s, "_cfg", lambda key, default: DEFAULTS.get(key, default))
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
    return cab, trades


def _drawdown_sells(trades, sid):
    return [t for t in trades if t["strategy"] == sid and t.get("reason") == "DRAWDOWN_LIMIT"]


def test_hold_and_timing_run_side_by_side_on_same_period(monkeypatch):
    df = _rise_then_crash("510300.SH")
    cab, _ = _run(monkeypatch, df, [Strategy00(), Strategy13()], max_drawdown_pct=0.0)
    hold_tx = cab.strategy_revenues["00"].transactions
    timing_tx = cab.strategy_revenues["13"].transactions
    # 两个策略各用一半资金、独立记账
    assert cab.strategy_initial_capital == pytest.approx(cab.initial_capital / 2)
    assert hold_tx[0]["direction"] == "BUY" and pd.Timestamp(hold_tx[0]["dt"]) == df["dt"].iloc[1]
    assert sum(1 for t in hold_tx if t["direction"] == "BUY") == 1
    # 择时策略在上涨中入场、在下跌中离场
    buys = [t for t in timing_tx if t["direction"] == "BUY"]
    sells = [t for t in timing_tx if t["direction"] == "SELL"]
    assert buys and sells
    assert pd.Timestamp(sells[-1]["dt"]) < df["dt"].iloc[-1]
    # 下跌段择时策略提前离场，期末现金多于持有到底的策略
    assert cab.strategy_revenues["13"].cash > cab.strategy_revenues["00"].cash


def test_drawdown_limit_liquidates_timing_but_not_hold(monkeypatch):
    df = _rise_then_gap_down("510300.SH")
    cab, trades = _run(monkeypatch, df, [Strategy00(), Strategy13()], max_drawdown_pct=0.06)
    assert _drawdown_sells(trades, "13")
    assert not _drawdown_sells(trades, "00")
    # 00 全程持有：只买一次，唯一的卖出是回测结束时的平仓
    hold_tx = cab.strategy_revenues["00"].transactions
    assert [t["direction"] for t in hold_tx] == ["BUY", "SELL"]
    assert pd.Timestamp(hold_tx[-1]["dt"]) == df["dt"].iloc[-1]


def test_drawdown_limit_still_applies_without_hold(monkeypatch):
    cab, trades = _run(monkeypatch, _rise_then_gap_down("510300.SH"), [Strategy13()], max_drawdown_pct=0.06)
    assert _drawdown_sells(trades, "13")


def test_hold_alone_is_never_liquidated_by_drawdown_limit(monkeypatch):
    cab, trades = _run(monkeypatch, _rise_then_gap_down("510300.SH"), [Strategy00()], max_drawdown_pct=0.06)
    assert not _drawdown_sells(trades, "00")
    assert [t["direction"] for t in cab.strategy_revenues["00"].transactions] == ["BUY", "SELL"]
