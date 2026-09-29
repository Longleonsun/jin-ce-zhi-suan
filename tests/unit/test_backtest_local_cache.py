"""本地缓存已覆盖回测区间时跳过连通性检查，断网也能回测。"""

import asyncio

import pandas as pd
import pytest

from src.core.backtest_cabinet import BacktestCabinet
from src.core.zhongshu_sheng import ZhongshuSheng
from src.ministries.hu_bu_revenue import HuBuRevenue
from src.strategies.implemented_strategies import Strategy00
from src.utils.yahoo_provider import YahooProvider

START, END = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-03-29")


def _daily(code="510300.SH"):
    dts = pd.bdate_range(START, END)
    c = pd.Series(range(len(dts)), dtype=float) * 0.01 + 4.0
    return pd.DataFrame({"code": code, "dt": dts, "open": c.values, "high": c.values * 1.01,
                         "low": c.values * 0.99, "close": c.values, "vol": 1e6, "amount": 1e8})


@pytest.fixture
def yahoo(tmp_path, monkeypatch):
    y = YahooProvider.__new__(YahooProvider)
    y._cache_enabled, y._cache_dir, y.last_error = True, str(tmp_path), ""
    _daily().to_csv(y._cache_file_path("510300.SH", "1d"), index=False)

    def _offline(*a, **k):
        raise ConnectionError("offline")

    monkeypatch.setattr(y, "_download", _offline)
    return y


# ---------- Yahoo 本地缓存覆盖判断 ----------

def test_covers_range_only_when_daily_cache_spans_request(yahoo):
    assert yahoo.covers_range("510300.SH", START, END, "D")
    assert yahoo.covers_range("510300.SH", START + pd.Timedelta(days=7), END - pd.Timedelta(days=7), "D")
    assert not yahoo.covers_range("510300.SH", START - pd.Timedelta(days=1), END, "D")
    assert not yahoo.covers_range("510300.SH", START, END + pd.Timedelta(days=1), "D")
    assert not yahoo.covers_range("510300.SH", START, END, "30min")  # 分钟线不落盘
    assert not yahoo.covers_range("159915.SZ", START, END, "D")


def test_covers_range_false_when_cache_disabled(yahoo):
    yahoo._cache_enabled = False
    assert not yahoo.covers_range("510300.SH", START, END, "D")


def test_covered_range_is_served_without_network(yahoo):
    df = yahoo.fetch_kline_data("510300.SH", START, END, "D")
    assert len(df) == len(_daily())


# ---------- 回测：断网时本地有数据照常跑，没有数据才报错 ----------

class _OfflineProvider:
    """模拟断网：连通性检查失败；covers 控制本地缓存是否覆盖回测区间。"""

    def __init__(self, df, covers):
        self.df, self.covers, self.last_error = df, covers, ""
        self.connectivity_checked = False

    def check_connectivity(self, code):
        self.connectivity_checked = True
        return False, "offline"

    def covers_range(self, code, start_time, end_time, interval="D"):
        return self.covers

    def fetch_kline_data(self, code, start_time, end_time, interval="D"):
        return self.df.copy() if self.covers else pd.DataFrame()


def _run(monkeypatch, provider):
    events = []

    async def _collect(event_type, data):
        events.append((event_type, data))

    df = provider.df
    cab = BacktestCabinet("510300.SH", strategy_ids=["00"], provider_override=provider,
                          provider_source_override="unit_stub", event_callback=_collect)
    cab.strategies = [Strategy00()]
    cab.secretariat = ZhongshuSheng(cab.strategies)
    cab.strategy_revenues = {"00": HuBuRevenue(cab.strategy_initial_capital)}

    async def _no_persist(*a, **k):
        return None

    monkeypatch.setattr(cab, "_persist_backtest_cache_to_db", _no_persist)
    monkeypatch.setattr(BacktestCabinet, "_tf_cache", {})
    asyncio.run(cab.run(start_date=df["dt"].min(), end_date=df["dt"].max()))
    return cab, events


def test_offline_backtest_runs_when_local_cache_covers_range(monkeypatch):
    provider = _OfflineProvider(_daily(), covers=True)
    cab, events = _run(monkeypatch, provider)
    assert not provider.connectivity_checked
    assert not any(t == "backtest_failed" for t, _ in events)
    assert cab.strategy_revenues["00"].transactions  # 正常成交
    assert any("跳过连通性检查" in str(d.get("msg", "")) for t, d in events if t == "backtest_flow")


def test_offline_backtest_fails_when_cache_missing(monkeypatch):
    provider = _OfflineProvider(_daily(), covers=False)
    cab, events = _run(monkeypatch, provider)
    assert provider.connectivity_checked
    assert any(t == "backtest_failed" for t, _ in events)
    assert not cab.strategy_revenues["00"].transactions


def test_memory_cache_hit_skips_connectivity_check(monkeypatch):
    provider = _OfflineProvider(_daily(), covers=False)
    df = provider.df
    key = ("510300.SH", "unit_stub", "D", df["dt"].min().strftime("%Y-%m-%d"), df["dt"].max().strftime("%Y-%m-%d"))
    events = []

    async def _collect(event_type, data):
        events.append((event_type, data))

    cab = BacktestCabinet("510300.SH", strategy_ids=["00"], provider_override=provider,
                          provider_source_override="unit_stub", event_callback=_collect)
    cab.strategies = [Strategy00()]
    cab.secretariat = ZhongshuSheng(cab.strategies)
    cab.strategy_revenues = {"00": HuBuRevenue(cab.strategy_initial_capital)}
    monkeypatch.setattr(BacktestCabinet, "_tf_cache", {key: df.copy()})
    asyncio.run(cab.run(start_date=df["dt"].min(), end_date=df["dt"].max()))
    assert not provider.connectivity_checked
    assert cab.strategy_revenues["00"].transactions
