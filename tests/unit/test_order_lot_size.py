"""下单数量按市场取整：A股 100 股（一手），美股 1 股。"""

import pandas as pd
import pytest

import src.core.zhongshu_sheng as zhongshu_mod
from src.core.zhongshu_sheng import ZhongshuSheng
from src.strategies.base_strategy import BaseStrategy
from src.strategies.implemented_strategies import BaseImplementedStrategy, Strategy00

CASH_PCT_CFG = {"order_qty_mode": "cash_pct", "order_cash_pct": 100}


def _bar(code, close):
    return {"code": code, "dt": pd.Timestamp("2024-01-02"), "open": close, "high": close,
            "low": close, "close": close, "vol": 1e6, "amount": 1e8}


def _ctx(sid, cash, price):
    return {"__by_strategy__": {sid: {"current_cash": cash, "last_price": price}}}


class _CashPctBuyer(BaseImplementedStrategy):
    """按 _qty() 下单；omit_qty=True 时不带 qty，走中书省兜底。"""

    def __init__(self, omit_qty=False):
        super().__init__("QT", "qty-probe", trigger_timeframe="D")
        self.omit_qty = omit_qty

    def on_bar(self, kline):
        signal = {"strategy_id": self.id, "code": kline["code"], "dt": kline["dt"],
                  "direction": "BUY", "price": kline["close"]}
        if not self.omit_qty:
            signal["qty"] = self._qty()
        return signal


class _NoQtyStrategy(BaseStrategy):
    """没有 _qty 的策略，中书省用全局配置兜底计算数量。"""

    def __init__(self):
        super().__init__("NQ")

    def set_backtest_context(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def on_bar(self, kline):
        return {"strategy_id": self.id, "code": kline["code"], "dt": kline["dt"],
                "direction": "BUY", "price": kline["close"]}


def _cash_pct_strategy(monkeypatch, omit_qty=False):
    s = _CashPctBuyer(omit_qty=omit_qty)
    monkeypatch.setattr(s, "_cfg", lambda key, default: CASH_PCT_CFG.get(key, default))
    return s


def _signal_qty(strategy, code, cash, price):
    signals = ZhongshuSheng([strategy]).generate_signals(
        _bar(code, price), strategy_context=_ctx(strategy.id, cash, price))
    return signals[0]["qty"] if signals else 0


@pytest.fixture
def cash_pct_fallback(monkeypatch):
    cfg = {"strategy_params.common.order_qty_mode": "cash_pct",
           "strategy_params.common.order_cash_pct": 100}
    monkeypatch.setattr(zhongshu_mod, "get_value", lambda path, default=None: cfg.get(path, default))


# ---------- A股：保持按一手（100 股）取整 ----------

@pytest.mark.parametrize("code", ["600000.SH", "002262.SZ", "300750.SZ"])
def test_cn_qty_rounds_to_board_lot(monkeypatch, code):
    # 100000 / 12.34 = 8103.7 股 -> 8100 股
    assert _signal_qty(_cash_pct_strategy(monkeypatch), code, 100_000, 12.34) == 8100


def test_cn_qty_below_one_lot_is_zero(monkeypatch):
    # 5000 / 100 = 50 股，不足一手，不下单
    assert _signal_qty(_cash_pct_strategy(monkeypatch), "600000.SH", 5_000, 100.0) == 0


def test_cn_fallback_qty_from_strategy(monkeypatch):
    assert _signal_qty(_cash_pct_strategy(monkeypatch, omit_qty=True), "600000.SH", 100_000, 12.34) == 8100


def test_cn_fallback_qty_from_global_config(cash_pct_fallback):
    assert _signal_qty(_NoQtyStrategy(), "600000.SH", 100_000, 12.34) == 8100


def test_cn_strategy00_buys_whole_lots():
    s = Strategy00()
    s.set_backtest_context(current_cash=100_000)
    assert s.on_bar(_bar("600000.SH", 12.34))["qty"] == 8100


# ---------- 美股：按 1 股取整 ----------

@pytest.mark.parametrize("code, price, expected", [
    ("AAPL", 230.0, 434),     # 修复前为 400
    ("AAPL", 1500.0, 66),     # 修复前为 0（漏单）
    ("AAPL.US", 1500.0, 66),
    ("BRK-B", 1500.0, 66),
])
def test_us_qty_rounds_to_single_share(monkeypatch, code, price, expected):
    assert _signal_qty(_cash_pct_strategy(monkeypatch), code, 100_000, price) == expected


def test_us_fallback_qty_from_strategy(monkeypatch):
    assert _signal_qty(_cash_pct_strategy(monkeypatch, omit_qty=True), "AAPL", 100_000, 1500.0) == 66


def test_us_fallback_qty_from_global_config(cash_pct_fallback):
    assert _signal_qty(_NoQtyStrategy(), "AAPL", 100_000, 1500.0) == 66


def test_us_strategy00_buys_single_shares():
    s = Strategy00()
    s.set_backtest_context(current_cash=100_000)
    assert s.on_bar(_bar("AAPL", 1500.0))["qty"] == 66


def test_qty_explicit_code_overrides_current_code(monkeypatch):
    s = _cash_pct_strategy(monkeypatch)
    s.set_backtest_context(current_cash=100_000, last_price=1500.0)
    s.current_code = "600000.SH"
    assert s._qty() == 0
    assert s._qty("AAPL") == 66
