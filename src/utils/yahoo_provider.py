import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from src.utils.config_loader import ConfigLoader
from src.utils.market_rules import market_profile

# Yahoo Finance intraday history limits (days)
_INTRADAY_LIMITS = {
    "1m": 7,
    "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60,
    "60m": 730, "1h": 730,
}

_INTERVAL_MINUTES = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90, "1h": 60}

# 日线缓存复权校验：增量拉取时回看的重叠天数，以及收盘价允许的相对误差
_ADJUST_CHECK_OVERLAP_DAYS = 10
_ADJUST_CHECK_TOLERANCE = 5e-4
# 缓存起点允许晚于请求起点的天数（覆盖节假日/周末），超过则整段重拉
_BACKFILL_TOLERANCE_DAYS = 7


class YahooProvider:
    """
    Yahoo Finance Data Provider — works globally, no API key required.
    Supports A-shares: SZ stocks use .SZ suffix, SH stocks use .SS suffix.
    Intraday support: 1h (730d history), 30m/15m/5m (60d), 1m (7d).
    """

    def __init__(self):
        cfg = ConfigLoader.reload()
        self.last_error = ""
        self._cache_enabled = bool(cfg.get("data_provider.local_cache_enabled", True))
        cache_dir = str(cfg.get("data_provider.local_cache_dir", "data/history/cache") or "data/history/cache")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(base_dir))
        self._cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(project_root, cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

    def _to_yahoo_symbol(self, code: str) -> str:
        c = str(code or "").strip().upper()
        if c.endswith(".SH"):
            return c[:-3] + ".SS"
        if c.endswith(".SZ"):
            return c
        if c.endswith(".US"):
            return c[:-3]
        if c.endswith(".O") or c.endswith(".N"):
            return c.rsplit(".", 1)[0]
        return c

    def _to_yfinance_interval(self, interval: str) -> str:
        """Map internal interval codes to yfinance interval strings."""
        x = str(interval).strip().lower()
        mapping = {
            "d": "1d", "1d": "1d", "day": "1d", "daily": "1d",
            "1min": "1m", "1m": "1m",
            "5min": "5m", "5m": "5m",
            "15min": "15m", "15m": "15m",
            "30min": "30m", "30m": "30m",
            "60min": "60m", "60m": "60m", "1h": "60m", "hour": "60m",
            "120min": "60m", "2h": "60m",
        }
        return mapping.get(x, "1d")

    def _cache_file_path(self, code, yf_interval):
        safe_code = str(code).upper().replace(".", "_")
        safe_iv = yf_interval.replace("/", "_")
        return os.path.join(self._cache_dir, f"yahoo_{safe_code}_{safe_iv}.csv")

    @staticmethod
    def _bar_end_labels(starts, code, minutes):
        """Yahoo 分钟级K线以开始时间标记，统一改为结束时间（与A股数据源及实盘重采样一致），
        并截断到所在交易时段的收盘（如美股 60m 最后一根 15:30 开始、16:00 结束）。
        开始时间不在任何交易时段内的K线（如A股 11:30 的零时长成交记录）返回 NaT，由调用方丢弃。"""
        sessions = market_profile(code)["sessions"]

        def _end(ts):
            hm = ts.hour * 60 + ts.minute
            end_hm = hm + minutes
            for start, stop in sessions:
                if start <= hm < stop:
                    return ts + pd.Timedelta(minutes=min(end_hm, stop) - hm)
            return pd.NaT

        return pd.to_datetime(starts.map(_end))

    def _normalize_raw(self, raw, code, yf_interval="1d"):
        """Flatten columns, rename to internal schema, strip timezone."""
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw.reset_index()
        for date_col in ("datetime", "date", "index"):
            if date_col in raw.columns:
                raw = raw.rename(columns={date_col: "dt"})
                break
        raw = raw.rename(columns={"volume": "vol"})
        raw["dt"] = pd.to_datetime(raw["dt"])
        minutes = _INTERVAL_MINUTES.get(yf_interval, 0)
        if minutes:
            raw["dt"] = self._bar_end_labels(raw["dt"], code, minutes)
            raw = raw.dropna(subset=["dt"])
            # 尚未走完的K线（结束时间晚于当前时间）不返回，避免策略基于残缺K线计算
            tz = raw["dt"].dt.tz
            now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()
            raw = raw[raw["dt"] <= now].copy()
        raw["dt"] = raw["dt"].dt.tz_localize(None)
        raw["code"] = code
        raw = raw.dropna(subset=["close", "open", "high", "low"])
        if "amount" not in raw.columns:
            raw["amount"] = raw["close"] * raw["vol"]
        cols = ["code", "dt", "open", "high", "low", "close", "vol", "amount"]
        return raw[[c for c in cols if c in raw.columns]]

    def _download(self, symbol, start_time, end_time, yf_interval):
        # auto_adjust=True：OHLC 均按前复权调整（最新价为实际价）
        return yf.download(
            symbol,
            start=start_time.strftime("%Y-%m-%d"),
            end=(end_time + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
        )

    @staticmethod
    def _same_adjust_basis(cached_df, fresh_df):
        """比较重叠区间的收盘价，判断缓存与新数据是否处于同一前复权基准。"""
        cached = cached_df[["dt", "close"]].copy()
        cached = cached[cached["dt"] < cached["dt"].max()]  # 最后一根可能是盘中未收盘数据，不参与比较
        merged = cached.merge(fresh_df[["dt", "close"]], on="dt", suffixes=("_cached", "_fresh"))
        merged = merged[(merged["close_cached"] > 0) & (merged["close_fresh"] > 0)]
        if merged.empty:
            return False
        ratio = merged["close_fresh"] / merged["close_cached"]
        return bool((ratio - 1.0).abs().max() <= _ADJUST_CHECK_TOLERANCE)

    def check_connectivity(self, code):
        try:
            symbol = self._to_yahoo_symbol(code)
            end = datetime.now()
            start = end - timedelta(days=10)
            df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return True, "ok"
            return False, "Yahoo Finance 返回空数据"
        except Exception as e:
            return False, f"Yahoo Finance 连通性检查异常: {e}"

    def fetch_daily_data(self, code, start_time, end_time):
        return self._fetch_data(code, start_time, end_time, "1d")

    def _load_cache(self, cache_path):
        if not (self._cache_enabled and os.path.exists(cache_path)):
            return pd.DataFrame()
        try:
            cached_df = pd.read_csv(cache_path)
            cached_df["dt"] = pd.to_datetime(cached_df["dt"])
            return cached_df.dropna(subset=["close", "open", "high", "low"])
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _cache_covers(cached_df, start_time, end_time):
        return not cached_df.empty and cached_df["dt"].min() <= start_time and cached_df["dt"].max() >= end_time

    def covers_range(self, code, start_time, end_time, interval="D"):
        """本地缓存是否已完整覆盖区间；覆盖时 fetch_kline_data 直接读缓存、不联网。仅日线落盘。"""
        yf_interval = self._to_yfinance_interval(interval)
        if yf_interval != "1d":
            return False
        return self._cache_covers(self._load_cache(self._cache_file_path(code, yf_interval)), start_time, end_time)

    def fetch_kline_data(self, code, start_time, end_time, interval="D"):
        yf_interval = self._to_yfinance_interval(interval)
        return self._fetch_data(code, start_time, end_time, yf_interval)

    def fetch_minute_data(self, code, start_time, end_time):
        return self._fetch_data(code, start_time, end_time, "1m")

    def _fetch_data(self, code, start_time, end_time, yf_interval):
        symbol = self._to_yahoo_symbol(code)
        cache_path = self._cache_file_path(code, yf_interval)
        limit_days = _INTRADAY_LIMITS.get(yf_interval, 36500)

        # Clamp start_time to Yahoo's history limit for intraday.
        # Use a 5-day safety buffer: Yahoo counts from the current moment, but
        # strftime truncates to midnight, so requesting exactly (limit-1) days
        # ago can land just outside the allowed window.
        if yf_interval != "1d":
            earliest = datetime.now() - timedelta(days=limit_days - 5)
            if start_time < earliest:
                start_time = earliest

        cached_df = self._load_cache(cache_path)
        if self._cache_covers(cached_df, start_time, end_time):
            return cached_df[(cached_df["dt"] >= start_time) & (cached_df["dt"] <= end_time)].copy()

        if not cached_df.empty and cached_df["dt"].min() > start_time + timedelta(days=_BACKFILL_TOLERANCE_DAYS):
            # 缓存起点晚于请求起点（例如先跑了短周期实盘预热），增量拉取只会向后补，需整段重拉
            cached_df = pd.DataFrame()
        if cached_df.empty:
            fetch_start = start_time
        elif yf_interval == "1d":
            # 日线增量拉取时回看一段重叠区间，用于校验复权基准是否变化
            fetch_start = cached_df["dt"].max() - timedelta(days=_ADJUST_CHECK_OVERLAP_DAYS)
        else:
            fetch_start = cached_df["dt"].max() + timedelta(seconds=1)
        if fetch_start > end_time:
            return cached_df[(cached_df["dt"] >= start_time) & (cached_df["dt"] <= end_time)].copy()

        try:
            raw = self._download(symbol, fetch_start, end_time, yf_interval)
            if raw is None or raw.empty:
                self.last_error = f"Yahoo {yf_interval} 数据为空 code={code}"
                return cached_df if not cached_df.empty else pd.DataFrame()

            raw = self._normalize_raw(raw, code, yf_interval)

            if not cached_df.empty and yf_interval == "1d" and not self._same_adjust_basis(cached_df, raw):
                # 分红/拆股后 Yahoo 会重算全部历史的前复权价，旧缓存与新数据混用会产生虚假跳空，整段重拉
                full_start = min(start_time, cached_df["dt"].min())
                full = self._download(symbol, full_start, end_time, yf_interval)
                if full is None or full.empty:
                    self.last_error = f"Yahoo 复权基准变化且重拉失败 code={code}"
                    return pd.DataFrame()
                raw = self._normalize_raw(full, code, yf_interval)
                cached_df = pd.DataFrame()

            if not cached_df.empty:
                cached_df = cached_df.dropna(subset=["close", "open", "high", "low"])
                raw = pd.concat([cached_df, raw], ignore_index=True)
            # 新拉取的数据优先（修正缓存中盘中未收盘的日线）
            raw = raw.drop_duplicates(subset=["dt"], keep="last").sort_values("dt").reset_index(drop=True)

            # Only cache daily data; intraday cache expires quickly
            if self._cache_enabled and yf_interval == "1d":
                raw.to_csv(cache_path, index=False, encoding="utf-8")

            self.last_error = ""
            return raw[(raw["dt"] >= start_time) & (raw["dt"] <= end_time)].copy()

        except Exception as e:
            self.last_error = f"Yahoo {yf_interval} 拉取失败 code={code} err={e}"
            return cached_df if not cached_df.empty else pd.DataFrame()

    def get_latest_bar(self, code, interval=None):
        """Get the most recent bar. interval=None/'D' → daily; else intraday (e.g. '60min')."""
        try:
            symbol = self._to_yahoo_symbol(code)
            is_daily = (not interval) or str(interval).upper() in ("D", "1D", "DAY", "DAILY", "1d")
            yf_interval = "1d" if is_daily else self._to_yfinance_interval(interval)

            if is_daily:
                end = datetime.now() + timedelta(days=1)  # end is exclusive in yfinance
                start = end - timedelta(days=8)
                raw = yf.download(
                    symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
            else:
                # 2 trading days covers current session regardless of timezone
                raw = yf.download(
                    symbol,
                    period="2d",
                    interval=yf_interval,
                    progress=False,
                    auto_adjust=True,
                )

            if raw is None or raw.empty:
                self.last_error = f"Yahoo 最新行情为空 code={code} interval={yf_interval}"
                return None

            raw = self._normalize_raw(raw, code, yf_interval)
            if raw.empty:
                self.last_error = f"Yahoo 最新行情无已完成K线 code={code} interval={yf_interval}"
                return None
            row = raw.iloc[-1]
            dt = pd.to_datetime(row["dt"]).to_pydatetime()
            self.last_error = ""
            return {
                "code": code,
                "dt": dt,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "vol": float(row["vol"]),
                "amount": float(row["amount"]),
            }
        except Exception as e:
            self.last_error = f"Yahoo get_latest_bar 失败 code={code} err={e}"
            return None
