"""回测报告：逐日净值口径的回撤、年化折算，以及与同期持有的对比（上涨区捕获率等）。"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.backtest_cabinet import BacktestCabinet
from src.core.zhongshu_sheng import ZhongshuSheng
from src.ministries.hu_bu_revenue import HuBuRevenue
from src.ministries.li_bu_rites import LiBuRites
from src.ministries.xing_bu_justice import XingBuJustice
from src.strategies.implemented_strategies import Strategy00


def _close(values, start="2024-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def _zigzag(n=300):
    """先涨后跌再涨，保证上涨区、下跌区都有足够天数。"""
    x = np.arange(n)
    return _close(10 + 3 * np.sin(x / 25) + x * 0.01)


# ---------- 与同期持有对比 ----------

def test_hold_strategy_matches_benchmark():
    close = _zigzag()
    b = LiBuRites().compute_benchmark_comparison(close * 1000, close)
    assert b["up_capture"] == pytest.approx(1.0)
    assert b["down_strategy_return"] == pytest.approx(b["down_bh_return"])
    assert b["bh_max_dd"] == pytest.approx(abs((close / close.cummax() - 1).min()))
    assert b["up_days"] > 20 and b["down_days"] > 20
    # MA50 形成之前（前 50 天）不计入任何区间
    assert b["up_days"] + b["down_days"] == len(close) - 50


def test_cash_strategy_captures_nothing():
    close = _zigzag()
    b = LiBuRites().compute_benchmark_comparison(pd.Series(1e6, index=close.index), close)
    assert b["up_capture"] == pytest.approx(0.0)
    assert b["down_strategy_return"] == 0.0


def test_regime_uses_previous_close_only():
    # 前 60 天横盘，第 61 天大涨站上均线：这一天的收益应计入下跌区（前一日收盘仍在均线下方）
    values = [10.0] * 60 + [12.0] + [12.0] * 5
    close = _close(values)
    b = LiBuRites().compute_benchmark_comparison(close, close)
    jump = 12.0 / 10.0 - 1
    assert b["down_bh_return"] == pytest.approx(jump)
    assert b["up_bh_return"] == pytest.approx(0.0)


def test_up_capture_absent_when_hold_not_profitable_in_up_regime():
    close = _close([10.0] * 80)
    b = LiBuRites().compute_benchmark_comparison(close, close)
    assert b["up_capture"] is None


# ---------- 报告口径 ----------

def test_report_drawdown_includes_unrealized_losses():
    """只在期末平仓的持有策略：旧口径按平仓盈亏算回撤为 0，逐日净值口径应反映持仓期间的回撤。"""
    close = _close([10, 11, 12, 8.4, 9, 10, 12.5])
    hu_bu = HuBuRevenue(1_000_000)
    hu_bu.transactions.append({"strategy_id": "00", "dt": close.index[-1], "direction": "SELL",
                               "pnl": 250_000.0, "amount": 1_250_000.0})
    rites = LiBuRites()
    old = rites.generate_report("00", hu_bu, XingBuJustice(), 1_000_000, close.index[0], close.index[-1])
    new = rites.generate_report("00", hu_bu, XingBuJustice(), 1_000_000, close.index[0], close.index[-1],
                                nav_series=close * 100_000, close_series=close)
    assert old["max_dd"] == pytest.approx(0.0)
    assert new["max_dd"] == pytest.approx(0.3)  # 12 → 8.4
    assert new["benchmark"]["bh_total_return"] == pytest.approx(0.25)


def test_annualization_uses_calendar_days():
    hu_bu = HuBuRevenue(1_000_000)
    start, end = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
    hu_bu.transactions.append({"strategy_id": "S", "dt": end, "direction": "SELL", "pnl": 210_000.0, "amount": 1.0})
    report = LiBuRites().generate_report("S", hu_bu, XingBuJustice(), 1_000_000, start, end)
    assert report["annualized_roi"] == pytest.approx(0.10, abs=2e-3)  # 两年累计 21% ≈ 年化 10%


# ---------- 端到端：回测报告带上逐日净值指标与持有对比 ----------

class _Provider:
    def __init__(self, df):
        self.df, self.last_error = df, ""

    def check_connectivity(self, code):
        return True, "ok"

    def fetch_kline_data(self, code, start_time, end_time, interval="D"):
        return self.df.copy()


def test_backtest_reports_carry_nav_metrics_and_benchmark(monkeypatch):
    close = _zigzag(200)
    c = close.to_numpy()
    df = pd.DataFrame({"code": "510300.SH", "dt": close.index, "open": c, "high": c * 1.01, "low": c * 0.99,
                       "close": c, "vol": 1e6, "amount": 1e8})
    events = []

    async def _collect(event_type, data):
        if event_type in ("backtest_strategy_report", "backtest_result"):
            events.append((event_type, data))

    cab = BacktestCabinet("510300.SH", strategy_ids=["00"], provider_override=_Provider(df),
                          provider_source_override="unit_stub", event_callback=_collect)
    cab.strategies = [Strategy00()]
    cab.secretariat = ZhongshuSheng(cab.strategies)
    cab.strategy_revenues = {"00": HuBuRevenue(cab.strategy_initial_capital)}

    async def _no_persist(*a, **k):
        return None

    monkeypatch.setattr(cab, "_persist_backtest_cache_to_db", _no_persist)
    monkeypatch.setattr(BacktestCabinet, "_tf_cache", {})
    asyncio.run(cab.run(start_date=df["dt"].min(), end_date=df["dt"].max()))

    report = next(d for t, d in events if t == "backtest_strategy_report")
    ranking = next(d for t, d in events if t == "backtest_result")["ranking"]
    json.dumps(ranking, allow_nan=False)  # 前端按 JSON 解析，不能出现 NaN
    b = report["benchmark"]
    # 00 号即持有：回撤与持有一致（从第二天开盘买入，略有差异），上涨区全部吃到
    assert abs(report["max_drawdown"]) == pytest.approx(b["bh_max_dd"], abs=0.03)
    assert abs(report["max_drawdown"]) > 0.2
    assert b["up_capture"] == pytest.approx(1.0, abs=0.05)
    assert ranking[0]["max_dd"] == pytest.approx(abs(report["max_drawdown"]))
    assert len(cab._daily_nav["00"]) == len(df)


# ---------- 批量回测结果列 ----------

def test_batch_result_columns_and_aliases_align():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import batch_backtest_runner as runner
    assert [key for _, key in runner.结果列定义] == runner.结果英文别名
    assert {"bh_annual_return", "up_capture", "down_bh_return"} <= set(runner.结果英文别名)
