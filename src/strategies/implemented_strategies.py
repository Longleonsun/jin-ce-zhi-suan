# src/strategies/implemented_strategies.py
from src.strategies.base_strategy import BaseStrategy
from src.utils.indicators import Indicators
import pandas as pd
import numpy as np
from src.utils.runtime_params import get_value
from src.utils.market_rules import lot_size_for_code

class BaseImplementedStrategy(BaseStrategy):
    """
    Base class for implemented strategies with common utilities.
    """
    def __init__(self, strategy_id, name, trigger_timeframe="1min"):
        super().__init__(strategy_id)
        self.name = name
        self.trigger_timeframe = trigger_timeframe
        self.bars_held = {} # Code -> Count of bars held
        self.entry_price = {} # Code -> Entry Price
        self.highest_high = {} # Code -> Highest High since entry
        self.trailing_stop_level = {} # Code -> Trailing Stop Price
        self.current_cash = 0.0
        self.available_cash = 0.0
        self.total_value = 0.0
        self.last_price = 0.0
        self.current_code = None # 当前处理的标的，由中书省在 on_bar 前写入

    def _lot_size(self, code=None):
        """下单单位：A股 100 股（一手），美股 1 股；未知标的按 A股处理。"""
        return lot_size_for_code(code or self.current_code)

    def update_holding_time(self, code):
        if code in self.positions and self.positions[code] > 0:
            self.bars_held[code] = self.bars_held.get(code, 0) + 1
        else:
            self.bars_held[code] = 0
            self.highest_high[code] = 0.0
            self.trailing_stop_level[code] = 0.0

    def check_max_holding_time(self, code, max_bars):
        if self.bars_held.get(code, 0) >= max_bars:
            return True
        return False
        
    def create_exit_signal(self, kline, qty, reason):
        return {
            'strategy_id': self.id,
            'code': kline['code'],
            'dt': kline['dt'],
            'direction': 'SELL',
            'price': kline['close'],
            'qty': qty,
            'reason': reason
        }

    def _cfg(self, key, default):
        own = get_value(f"strategy_params.{self.id}.{key}", None)
        if own is not None:
            return own
        common = get_value(f"strategy_params.common.{key}", None)
        return common if common is not None else default

    def _qty(self, code=None):
        mode = str(self._cfg("order_qty_mode", "fixed")).strip().lower()
        fixed_qty = int(float(self._cfg("order_qty", 1000)))
        if mode != "cash_pct":
            return max(0, fixed_qty)
        cash = float(
            getattr(self, "current_cash", None)
            if getattr(self, "current_cash", None) is not None
            else getattr(self, "available_cash", getattr(self, "cash", 0.0))
        )
        pct = float(self._cfg("order_cash_pct", 0.1))
        price = float(getattr(self, "last_price", 0.0) or 0.0)
        if pct > 1:
            pct = pct / 100.0
        pct = max(0.0, min(1.0, pct))
        if cash <= 0 or price <= 0 or pct <= 0:
            return 0
        raw_qty = int((cash * pct) // price)
        lot_size = self._lot_size(code)
        lot_qty = (raw_qty // lot_size) * lot_size
        return max(0, lot_qty)

    def set_backtest_context(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if "current_cash" in kwargs and "available_cash" not in kwargs:
            self.available_cash = kwargs.get("current_cash")
        if "available_cash" in kwargs and "current_cash" not in kwargs:
            self.current_cash = kwargs.get("available_cash")
        if "total_value" not in kwargs:
            self.total_value = float(self.current_cash or self.available_cash or 0.0)

class Strategy00(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("00", "长期持有一次买入", trigger_timeframe="D")
        self.entered = {}
        self.final_bar_dt = None

    def on_bar(self, kline):
        code = kline['code']
        qty = int(self.positions.get(code, 0))
        current_dt = pd.to_datetime(kline['dt'])
        if qty > 0:
            self.entered[code] = True

        if qty <= 0 and not self.entered.get(code, False):
            price = float(kline.get('close', 0.0))
            cash = float(getattr(self, "current_cash", 0.0) or 0.0)
            raw_qty = int(cash // price) if price > 0 else 0
            lot = self._lot_size(code)
            buy_qty = (raw_qty // lot) * lot
            if buy_qty <= 0:
                return None
            return {
                'strategy_id': self.id,
                'code': code,
                'dt': kline['dt'],
                'direction': 'BUY',
                'price': kline['close'],
                'qty': buy_qty,
                'stop_loss': 0.0,
                'take_profit': None
            }

        if qty > 0 and self.final_bar_dt is not None and current_dt >= pd.to_datetime(self.final_bar_dt):
            return self.create_exit_signal(kline, qty, "Backtest Last Bar Exit")

        return None

# Strategy 01: 三周期共振波段策略
class Strategy01(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("01", "三周期共振波段", trigger_timeframe="60min")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        self.update_holding_time(code)
        
        if code not in self.history: self.history[code] = pd.DataFrame()
        new_row = pd.DataFrame([kline])
        self.history[code] = pd.concat([self.history[code], new_row], ignore_index=True)
        
        if len(self.history[code]) > 30000:
             self.history[code] = self.history[code].iloc[-30000:]
             
        df = self.history[code]
        if len(df) < int(self._cfg("min_history_bars", 500)): return None 
        
        qty = self.positions.get(code, 0)
        
        # Optimization: Use subsets for resampling
        df_d_subset = df.iloc[-6000:]
        df_d = Indicators.resample(df_d_subset, 'D')
        if len(df_d) < 20: return None
        df_d['ma10'] = Indicators.MA(df_d['close'], 10)
        df_d['ma20'] = Indicators.MA(df_d['close'], 20)
        curr_d = df_d.iloc[-1]
        
        if qty > 0:
            if curr_d['close'] < curr_d['ma20']:
                return self.create_exit_signal(kline, qty, "Daily MA20 Breakdown")
            return None

        # Weekly
        df_w = Indicators.resample(df, 'W')
        if len(df_w) < 20: return None
        df_w['ma20'] = Indicators.MA(df_w['close'], 20)

        # 60 Min
        df_recent = df.iloc[-2000:]
        df_60m = Indicators.resample(df_recent, '60min')
        if len(df_60m) < 35: return None
        df_60m['dif'], df_60m['dea'], df_60m['macd'] = Indicators.MACD(df_60m['close'])
        
        curr_60m = df_60m.iloc[-1]
        prev_60m = df_60m.iloc[-2]
            
        # Entry
        if len(df_w) >= 2 and df_w.iloc[-1]['ma20'] > df_w.iloc[-2]['ma20']:
            if abs(curr_d['low'] - curr_d['ma10']) / curr_d['ma10'] < float(self._cfg("daily_ma10_tolerance", 0.02)):
                if curr_60m['dif'] > curr_60m['dea'] and prev_60m['dif'] <= prev_60m['dea']:
                    vol_ma5 = df_60m['vol'].rolling(int(self._cfg("volume_ma_window", 5))).mean().iloc[-1]
                    if curr_60m['vol'] > vol_ma5:
                        return {
                            'strategy_id': self.id,
                            'code': code,
                            'dt': kline['dt'],
                            'direction': 'BUY',
                            'price': kline['close'],
                            'qty': self._qty(),
                            'stop_loss': kline['close'] * (1 - float(self._cfg("stop_loss_pct", 0.03))),
                            'take_profit': None
                        }
        return None

# Strategy 02: 短线弱转强烂板战法
class Strategy02(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("02", "短线弱转强烂板", trigger_timeframe="1min")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        self.update_holding_time(code)
        if code not in self.history: self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(500)
        
        qty = self.positions.get(code, 0)
        if qty > 0:
            if self.check_max_holding_time(code, int(self._cfg("max_hold_bars", 240))): 
                return self.create_exit_signal(kline, qty, "Next Day Exit")
            return None
        return None

# Strategy 03: ETF行业轮动
class Strategy03(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("03", "ETF行业轮动", trigger_timeframe="D")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        if code not in self.history: self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(2400)
        df = self.history[code]
        
        if len(df) < 240: return None
        
        df_d = Indicators.resample(df, 'D')
        if len(df_d) < 10: return None
        df_d['ma10'] = Indicators.MA(df_d['close'], 10)
        curr_d = df_d.iloc[-1]
        
        qty = self.positions.get(code, 0)
        if qty > 0:
            if curr_d['close'] < curr_d['ma10']:
                return self.create_exit_signal(kline, qty, "Break MA10")
            return None
            
        if curr_d['close'] > curr_d['ma10']:
             return {
                'strategy_id': self.id,
                'code': code,
                'dt': kline['dt'],
                'direction': 'BUY',
                'price': kline['close'],
                'qty': self._qty(),
                'stop_loss': kline['close'] * (1 - float(self._cfg("stop_loss_pct", 0.05))),
                'take_profit': None
            }
        return None

# Strategy 04: 龙头首阴反包
class Strategy04(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("04", "龙头首阴反包", trigger_timeframe="D")
    def on_bar(self, kline):
        return None

# Strategy 05: 3N法则主升浪
class Strategy05(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("05", "3N法则主升浪", trigger_timeframe="30min")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        self.update_holding_time(code)
        if code not in self.history: self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(5000)
        df = self.history[code]
        
        qty = self.positions.get(code, 0)
        
        df_30m = Indicators.resample(df, '30min')
        if len(df_30m) < int(self._cfg("min_30m_bars", 20)): return None
        df_30m['dif'], df_30m['dea'], df_30m['macd'] = Indicators.MACD(df_30m['close'])
        curr = df_30m.iloc[-1]
        
        if qty > 0:
             if self.check_max_holding_time(code, int(self._cfg("max_hold_bars", 1200))): 
                 return self.create_exit_signal(kline, qty, "Time 5 Days")
             return None
             
        if curr['dif'] > curr['dea'] and curr['dif'] > 0:
            if curr['vol'] > df_30m['vol'].rolling(int(self._cfg("volume_ma_window", 5))).mean().iloc[-1] * float(self._cfg("volume_multiple", 1.5)):
                 return {
                    'strategy_id': self.id,
                    'code': code,
                    'dt': kline['dt'],
                    'direction': 'BUY',
                    'price': kline['close'],
                    'qty': self._qty(),
                    'stop_loss': kline['close'] * (1 - float(self._cfg("stop_loss_pct", 0.05))),
                    'take_profit': kline['close'] * (1 + float(self._cfg("take_profit_pct", 0.10)))
                }
        return None

# Strategy 06: 海豚交易法
class Strategy06(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("06", "海豚交易法", trigger_timeframe="1min")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        self.update_holding_time(code)
        if code not in self.history: self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(500)
        df = self.history[code]
        if len(df) < int(self._cfg("min_history_bars", 50)): return None
        
        # MA26 for Trend
        ma_period = int(self._cfg("ma_period", 26))
        df['ma26'] = Indicators.MA(df['close'], ma_period)
        # MACD
        df['dif'], df['dea'], df['macd'] = Indicators.MACD(df['close'])
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        qty = self.positions.get(code, 0)
        
        # Trailing Stop Logic (1% below High)
        if qty > 0:
            if curr['high'] > self.highest_high.get(code, 0.0):
                self.highest_high[code] = curr['high']
                # Update Trailing Stop: 1% below new high
                self.trailing_stop_level[code] = self.highest_high[code] * (1 - float(self._cfg("trailing_stop_pct", 0.01)))
            
            # Check Trailing Stop
            if curr['low'] <= self.trailing_stop_level.get(code, 0.0):
                return self.create_exit_signal(kline, qty, f"Trailing Stop (High {self.highest_high[code]:.2f})")
                
            return None
            
        # Entry Long
        # Price > MA26 + MACD Gold Cross above Zero
        if curr['close'] > curr['ma26']:
            # Gold Cross: DIF > DEA now, DIF <= DEA before
            if curr['dif'] > curr['dea'] and prev['dif'] <= prev['dea']:
                # "Water Top" (Above Zero)
                if curr['dif'] > 0 and curr['dea'] > 0:
                    self.highest_high[code] = curr['close']
                    self.trailing_stop_level[code] = curr['close'] * (1 - float(self._cfg("trailing_stop_pct", 0.01)))
                    return {
                        'strategy_id': self.id,
                        'code': code,
                        'dt': kline['dt'],
                        'direction': 'BUY',
                        'price': kline['close'],
                        'qty': self._qty(),
                        'stop_loss': kline['close'] * (1 - float(self._cfg("stop_loss_pct", 0.01))),
                        'take_profit': None # Trailing Stop only
                    }
        return None

# Strategy 07: 跳空交易系统
class Strategy07(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("07", "跳空交易系统", trigger_timeframe="15min")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        self.update_holding_time(code)
        if code not in self.history: self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(500)
        df = self.history[code]
        
        # Use 15min timeframe as per description
        df_15m = Indicators.resample(df, '15min')
        if len(df_15m) < int(self._cfg("min_15m_bars", 20)): return None
        
        curr = df_15m.iloc[-1]
        prev = df_15m.iloc[-2]
        
        qty = self.positions.get(code, 0)
        
        # Trailing Stop Logic (Similar to Dolphin: 1% below High)
        if qty > 0:
            # Check Stop
            current_price = kline['close']
            if current_price > self.highest_high.get(code, 0.0):
                self.highest_high[code] = current_price
                self.trailing_stop_level[code] = self.highest_high[code] * (1 - float(self._cfg("trailing_stop_pct", 0.01)))
            
            if current_price <= self.trailing_stop_level.get(code, 0.0):
                 return self.create_exit_signal(kline, qty, "Trailing Stop")
            return None
            
        # Entry Long
        # Gap Down 0.2% + Yang Line
        if curr['open'] < prev['close'] * float(self._cfg("gap_down_multiplier", 0.998)):
            if curr['close'] > curr['open']: # Yang Line
                self.highest_high[code] = curr['close']
                self.trailing_stop_level[code] = curr['close'] * (1 - float(self._cfg("trailing_stop_pct", 0.01)))
                return {
                    'strategy_id': self.id,
                    'code': code,
                    'dt': kline['dt'],
                    'direction': 'BUY',
                    'price': kline['close'],
                    'qty': self._qty(),
                    'stop_loss': kline['close'] * (1 - float(self._cfg("stop_loss_pct", 0.01))),
                    'take_profit': None
                }
        return None

# Strategy 08: 神奇九转 (Magic 9)
class Strategy08(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("08", "神奇九转", trigger_timeframe="1min")
        self.history = {}

    def on_bar(self, kline):
        code = kline['code']
        self.update_holding_time(code)
        
        # History Management
        if code not in self.history: self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(500)
        df = self.history[code]
        
        # Need enough data: 9 bars + 4 lag = 13 minimum
        if len(df) < int(self._cfg("min_history_bars", 13)): return None
        
        qty = self.positions.get(code, 0)
        curr_close = kline['close']
        
        # --- Exit Logic: Trailing Stop (1%) ---
        if qty > 0:
            # Update Highest High
            if kline['high'] > self.highest_high.get(code, 0.0):
                self.highest_high[code] = kline['high']
                self.trailing_stop_level[code] = self.highest_high[code] * (1 - float(self._cfg("trailing_stop_pct", 0.01)))
            
            # Check Stop
            if curr_close <= self.trailing_stop_level.get(code, 0.0):
                 return self.create_exit_signal(kline, qty, f"Trailing Stop (High {self.highest_high[code]:.2f})")
            return None

        # --- Entry Logic: Magic 9 (Buy Setup) ---
        # Condition: 9 consecutive bars where Close > Close[i-4]
        # Check last 9 bars
        
        # Get close prices as numpy array for speed
        closes = df['close'].values
        
        is_setup = True
        for i in range(1, 10): # 1 to 9 (checking last 9 bars)
            # Index from end: -i
            # closes[-i] vs closes[-i-4]
            if not (closes[-i] > closes[-i-4]):
                is_setup = False
                break
        
        # Specific Validation from Rules:
        # 1st (-9) > Prev 4 (-13) (Already checked in loop i=9)
        # 5th (-5) > 1st (-9)
        # 9th (-1) > 5th (-5)
        if is_setup:
            price_1st = closes[-9]
            price_5th = closes[-5]
            price_9th = closes[-1]
            
            if price_5th > price_1st and price_9th > price_5th:
                 # Initialize Trailing Stop state
                 self.highest_high[code] = curr_close
                 self.trailing_stop_level[code] = curr_close * (1 - float(self._cfg("trailing_stop_pct", 0.01)))
                 
                 return {
                    'strategy_id': self.id,
                    'code': code,
                    'dt': kline['dt'],
                    'direction': 'BUY',
                    'price': curr_close,
                    'qty': self._qty(),
                    'stop_loss': curr_close * (1 - float(self._cfg("stop_loss_pct", 0.01))),
                    'take_profit': None
                }
            
        return None

class Strategy09(BaseImplementedStrategy):
    def __init__(self):
        super().__init__("09", "箱体降本策略", trigger_timeframe="D")
        self.history = {}
        self.strategy_active = {}

    def on_bar(self, kline):
        code = kline['code']
        if code not in self.history:
            self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(60000)
        df = self.history[code]
        box_period = int(self._cfg("box_period", 240))
        rsi_period = int(self._cfg("rsi_period", 14))
        if len(df) < max(800, box_period * 3):
            return None
        df_d = Indicators.resample(df, 'D')
        if len(df_d) < box_period + 1:
            return None
        close = df_d['close']
        high = df_d['high']
        low = df_d['low']
        rsi_series = Indicators.RSI(close, rsi_period)
        c = float(close.iloc[-1])
        top = float(high.iloc[-(box_period + 1):-1].max())
        bot = float(low.iloc[-(box_period + 1):-1].min())
        if top <= bot:
            return None
        box_h = top - bot
        break_pct = float(self._cfg("break_pct", 0.05))
        rsi_oversold = float(self._cfg("rsi_oversold", 30))
        rsi_overbought = float(self._cfg("rsi_overbought", 70))
        base_build_rsi = float(self._cfg("base_build_rsi", 35))
        buy_zone_ratio = float(self._cfg("buy_zone_ratio", 0.15))
        sell_zone_ratio = float(self._cfg("sell_zone_ratio", 0.15))
        lot = self._lot_size(code)
        base_qty = int(self._cfg("base_order_qty", max(lot, self._qty(code))))
        dynamic_qty = int(self._cfg("dynamic_order_qty", max(lot, int(base_qty * 0.35))))
        max_dynamic_qty = int(self._cfg("max_dynamic_qty", dynamic_qty * 3))
        stop_loss_pct = float(self._cfg("stop_loss_pct", 0.05))
        if code not in self.strategy_active:
            self.strategy_active[code] = True
        if not self.strategy_active.get(code, True):
            return None
        qty = int(self.positions.get(code, 0))
        curr_rsi = float(rsi_series.iloc[-1])
        if c < bot * (1 - break_pct):
            self.strategy_active[code] = False
            if qty > 0:
                return self.create_exit_signal(kline, qty, "Box Breakdown Exit")
            return None
        if c > top * (1 + break_pct):
            self.strategy_active[code] = False
            return None
        mid = (top + bot) / 2.0
        if qty < base_qty:
            if bot <= c <= mid and curr_rsi < base_build_rsi:
                buy_qty = max(0, base_qty - qty)
                if buy_qty > 0:
                    return {
                        'strategy_id': self.id,
                        'code': code,
                        'dt': kline['dt'],
                        'direction': 'BUY',
                        'price': c,
                        'qty': buy_qty,
                        'stop_loss': c * (1 - stop_loss_pct),
                        'take_profit': None
                    }
            return None
        if bot <= c <= bot + box_h * buy_zone_ratio and curr_rsi <= rsi_oversold:
            cap_qty = max(0, base_qty + max_dynamic_qty - qty)
            buy_qty = min(dynamic_qty, cap_qty)
            if buy_qty > 0:
                return {
                    'strategy_id': self.id,
                    'code': code,
                    'dt': kline['dt'],
                    'direction': 'BUY',
                    'price': c,
                    'qty': buy_qty,
                    'stop_loss': c * (1 - stop_loss_pct),
                    'take_profit': None
                }
        if top - box_h * sell_zone_ratio <= c <= top and curr_rsi >= rsi_overbought:
            sell_qty = min(dynamic_qty, max(0, qty - base_qty))
            if sell_qty > 0:
                return self.create_exit_signal(kline, sell_qty, "Box Dynamic Rebalance")
        return None


class Strategy10(BaseImplementedStrategy):
    """内置选股示例策略：主板强势回撤（日线）。"""

    def __init__(self):
        # 固定为内置策略ID=10，便于策略管理器稳定展示。
        super().__init__("10", "选股示例-主板强势回撤", trigger_timeframe="D")
        # 为每个股票维护历史K线与买入日，支持 T+1 校验。
        self.history = {}
        self.last_buy_day = {}
        self.entry_price_local = {}

    def _is_main_board(self, code):
        # 仅做A股主板，避免创业板/科创板涨跌幅规则差异影响演示口径。
        c = str(code or "").split(".", 1)[0].strip().upper()
        return c.startswith(("600", "601", "603", "605", "000", "001", "002"))

    def _limit_pct(self, kline):
        # 统一涨跌幅计算，供涨跌停近似判断复用。
        pre_close = float(kline.get("pre_close", 0.0) or 0.0)
        close = float(kline.get("close", 0.0) or 0.0)
        if pre_close <= 0:
            return 0.0
        return (close - pre_close) / pre_close * 100.0

    def _is_limit_up(self, kline):
        # A股约束：涨停不追。
        pct = self._limit_pct(kline)
        return abs(pct - 10.0) < 0.1 or abs(pct - 20.0) < 0.1

    def _is_limit_down(self, kline):
        # A股约束：跌停不卖。
        pct = self._limit_pct(kline)
        return abs(pct + 10.0) < 0.1 or abs(pct + 20.0) < 0.1

    def _same_day(self, code, dt_value):
        # 标准化日期字符串用于 T+1 判断。
        day_text = str(pd.to_datetime(dt_value, errors="coerce").strftime("%Y-%m-%d"))
        return self.last_buy_day.get(code) == day_text, day_text

    def on_bar(self, kline):
        # 读取并校验标的代码。
        code = str(kline.get("code", "") or "").strip()
        if not code:
            return None
        if not self._is_main_board(code):
            return None

        # 维护历史窗口，900根足够支撑均线与性能。
        if code not in self.history:
            self.history[code] = pd.DataFrame()
        self.history[code] = pd.concat([self.history[code], pd.DataFrame([kline])], ignore_index=True).tail(900)
        df = self.history[code]
        if len(df) < 30:
            return None

        # 计算核心特征：MA5/MA20/近5日涨幅。
        close = df["close"].astype(float)
        ma5 = Indicators.MA(close, 5)
        ma20 = Indicators.MA(close, 20)
        if len(ma5) < 2 or len(ma20) < 2:
            return None
        curr_close = float(kline.get("close", 0.0) or 0.0)
        if curr_close <= 0:
            return None
        old_close = float(close.iloc[-6]) if len(close) >= 6 else curr_close
        change_5d = ((curr_close - old_close) / old_close * 100.0) if old_close > 0 else 0.0
        qty = int(self.positions.get(code, 0) or 0)

        # 可由运行参数覆盖的风控参数。
        stop_loss_pct = float(self._cfg("stop_loss_pct", 0.03))
        take_profit_pct = float(self._cfg("take_profit_pct", 0.08))
        max_hold_bars = int(self._cfg("max_hold_bars", 15))

        # 入场：空仓 + 趋势成立 + 近5日动量区间 + 非涨停。
        if qty <= 0:
            trend_ok = float(ma5.iloc[-1]) > float(ma20.iloc[-1]) and curr_close > float(ma20.iloc[-1])
            momentum_ok = 2.0 <= change_5d <= 25.0
            if trend_ok and momentum_ok and (not self._is_limit_up(kline)):
                buy_qty = int(self._qty(code))
                if buy_qty <= 0:
                    buy_qty = self._lot_size(code)
                _same, day_text = self._same_day(code, kline.get("dt"))
                self.last_buy_day[code] = day_text
                self.entry_price_local[code] = curr_close
                return {
                    "strategy_id": self.id,
                    "code": code,
                    "dt": kline["dt"],
                    "direction": "BUY",
                    "price": curr_close,
                    "qty": buy_qty,
                    "stop_loss": curr_close * (1 - stop_loss_pct),
                    "take_profit": curr_close * (1 + take_profit_pct),
                }
            return None

        # T+1：当日买入不可卖。
        same_day, _day_text = self._same_day(code, kline.get("dt"))
        if same_day:
            return None
        # 跌停日不可卖，等待下一交易日。
        if self._is_limit_down(kline):
            return None

        # 出场：死叉 / 止损 / 止盈 / 持仓超时。
        death_cross = float(ma5.iloc[-2]) >= float(ma20.iloc[-2]) and float(ma5.iloc[-1]) < float(ma20.iloc[-1])
        entry_price = float(self.entry_price_local.get(code, curr_close) or curr_close)
        stop_loss_hit = curr_close <= entry_price * (1 - stop_loss_pct)
        take_profit_hit = curr_close >= entry_price * (1 + take_profit_pct)
        self.update_holding_time(code)
        timeout_exit = self.check_max_holding_time(code, max_hold_bars)
        if death_cross or stop_loss_hit or take_profit_hit or timeout_exit:
            reason = []
            if death_cross:
                reason.append("MA Death Cross")
            if stop_loss_hit:
                reason.append("Stop Loss")
            if take_profit_hit:
                reason.append("Take Profit")
            if timeout_exit:
                reason.append("Time Exit")
            return self.create_exit_signal(kline, qty, " | ".join(reason) if reason else "Rule Exit")
        return None


class Strategy11(BaseImplementedStrategy):
    """MACD + KDJ + 成交量共振策略，面向 Yahoo 日线数据，兼容A股/美股。"""

    def __init__(self):
        super().__init__("11", "MACD-KDJ-成交量共振", trigger_timeframe="D")
        self._init_state()

    def _init_state(self):
        self.history = {}
        self.entry_price_local = {}  # Code -> 入场成本价（优先取引擎持仓成本）
        self.highest_close = {}      # Code -> 入场以来最高收盘价
        self.bar_seq = {}            # Code -> 已处理K线计数
        self.last_exit_seq = {}      # Code -> 最近一次平仓时的K线序号
        self.position_meta = {}      # Code -> {"avg_price", "entry_day"}，由引擎每根K线注入

    def update_position(self, code, qty):
        prev_qty = int(self.positions.get(code, 0) or 0)
        super().update_position(code, qty)
        qty = int(qty or 0)
        if qty > 0 and prev_qty <= 0:
            fill_price = float(getattr(self, "last_price", 0.0) or 0.0)
            if fill_price > 0:
                self.entry_price_local[code] = fill_price
                self.highest_close[code] = fill_price
        elif qty <= 0 and prev_qty > 0:
            self.entry_price_local.pop(code, None)
            self.highest_close.pop(code, None)
            self.last_exit_seq[code] = self.bar_seq.get(code, 0)

    def _append_history(self, code, kline):
        row = pd.DataFrame([dict(kline)])
        df = self.history.get(code)
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.history[code] = row
            return
        # 同一根K线重复推送（实盘预热/盘中刷新）时覆盖而非追加
        if "dt" in df.columns and pd.to_datetime(df["dt"].iloc[-1]) == pd.to_datetime(kline.get("dt")):
            df = df.iloc[:-1]
        self.history[code] = pd.concat([df, row], ignore_index=True).tail(360)

    def _buy_qty(self, code, price):
        """按市场下单单位取整，并限制单标的仓位不超过总资产的 max_position_pct。"""
        if price <= 0:
            return 0
        lot = lot_size_for_code(code)
        cash = float(getattr(self, "current_cash", getattr(self, "available_cash", 0.0)) or 0.0)
        total = float(getattr(self, "total_value", 0.0) or 0.0) or cash
        cap_pct = float(self._cfg("max_position_pct", 0.2))
        if cap_pct > 1:
            cap_pct = cap_pct / 100.0
        budget = min(cash, total * cap_pct) if cap_pct > 0 else cash
        mode = str(self._cfg("order_qty_mode", "fixed")).strip().lower()
        if mode == "cash_pct":
            pct = float(self._cfg("order_cash_pct", 0.1))
            if pct > 1:
                pct = pct / 100.0
            budget = min(budget, cash * max(0.0, min(1.0, pct)))
            raw_qty = int(budget // price)
        else:
            raw_qty = min(int(float(self._cfg("order_qty", 1000))), int(budget // price))
        return max(0, (raw_qty // lot) * lot)

    def _volume_baseline(self, df, vol):
        """放量基准：日线用近 N 根均量；分钟级用过去 N 个交易日同一时段的均量。
        盘中成交量集中在开盘和收盘时段，直接和相邻K线均量比，放量条件会退化成"是否开盘/收盘那根"。"""
        if self.trigger_timeframe == "D":
            ma = vol.rolling(int(self._cfg("volume_ma_window", 20))).mean().iloc[-1]
            return 0.0 if pd.isna(ma) else float(ma)
        days = int(self._cfg("volume_slot_days", 10))
        slot = pd.to_datetime(df["dt"], errors="coerce").dt.strftime("%H:%M")
        same_slot = vol[slot == slot.iloc[-1]].iloc[:-1].tail(days)
        if days <= 0 or len(same_slot) < days:
            return 0.0
        return float(same_slot.mean())

    def _restore_entry_state(self, code, df, curr_close):
        """重启后本地状态丢失时，用引擎持仓成本与买入日重建入场价、最高收盘价和持仓K线数。"""
        meta = self.position_meta.get(code) if isinstance(self.position_meta, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        avg_price = float(meta.get("avg_price", 0.0) or 0.0)
        if avg_price > 0:
            self.entry_price_local[code] = avg_price
        elif code not in self.entry_price_local:
            self.entry_price_local[code] = curr_close
        if code in self.highest_close:
            return
        entry_price = float(self.entry_price_local[code])
        entry_day = str(meta.get("entry_day", "") or "").strip()
        since_entry = pd.DataFrame()
        if entry_day and "dt" in df.columns:
            dts = pd.to_datetime(df["dt"], errors="coerce")
            since_entry = df[dts.dt.normalize() >= pd.Timestamp(entry_day)]
        if not since_entry.empty:
            self.highest_close[code] = max(entry_price, float(pd.to_numeric(since_entry["close"], errors="coerce").max()))
            self.bars_held[code] = max(int(self.bars_held.get(code, 0) or 0), len(since_entry) - 1)
        else:
            self.highest_close[code] = max(entry_price, curr_close)

    def on_bar(self, kline):
        code = str(kline.get("code", "") or "").strip()
        if not code:
            return None
        self.update_holding_time(code)
        self._append_history(code, kline)
        self.bar_seq[code] = self.bar_seq.get(code, 0) + 1
        df = self.history[code]

        min_bars = max(2, int(self._cfg("min_history_bars", 60)))
        if len(df) < min_bars:
            return None

        close = pd.to_numeric(df["close"], errors="coerce")
        high = pd.to_numeric(df["high"], errors="coerce")
        low = pd.to_numeric(df["low"], errors="coerce")
        vol = pd.to_numeric(df["vol"], errors="coerce").fillna(0.0)
        dif, dea, macd = Indicators.MACD(close)
        k, d, j = Indicators.KDJ(high, low, close)
        ma20 = Indicators.MA(close, 20)
        curr_close = float(close.iloc[-1])
        if curr_close <= 0:
            return None

        qty = int(self.positions.get(code, 0) or 0)
        stop_loss_pct = float(self._cfg("stop_loss_pct", 0.08))
        take_profit_pct = float(self._cfg("take_profit_pct", 0.18))
        trailing_stop_pct = float(self._cfg("trailing_stop_pct", 0.10))
        max_hold_bars = int(self._cfg("max_hold_bars", 80))

        dif_now, dif_prev = float(dif.iloc[-1]), float(dif.iloc[-2])
        dea_now, dea_prev = float(dea.iloc[-1]), float(dea.iloc[-2])
        k_now, k_prev = float(k.iloc[-1]), float(k.iloc[-2])
        d_now, d_prev = float(d.iloc[-1]), float(d.iloc[-2])
        j_now = float(j.iloc[-1])
        ma20_now = float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else None
        trend_ok = ma20_now is not None and curr_close > ma20_now

        if qty <= 0:
            cooldown = int(self._cfg("reentry_cooldown_bars", 3))
            last_exit = self.last_exit_seq.get(code)
            if last_exit is not None and self.bar_seq[code] - last_exit <= cooldown:
                return None
            macd_gold = dif_now > dea_now and dif_prev <= dea_prev
            # 零轴上方多头区间：DIF、DEA 均在零轴上方且 DIF > DEA（MACD 柱 > 0 与 DIF > DEA 等价）
            macd_bull = dif_now > dea_now and dea_now > 0 and float(macd.iloc[-1]) > 0
            kdj_gold = k_now > d_now and k_prev <= d_prev
            # 强势回升：K > D 且 J > K，且 K 值较上一根抬升（J > K 与 K > D 等价，需 K 抬升才构成"回升"）
            kdj_strong = k_now > d_now and j_now > k_now and k_now > k_prev
            vol_base = self._volume_baseline(df, vol)
            volume_ok = vol_base > 0 and float(vol.iloc[-1]) > vol_base * float(self._cfg("volume_multiple", 1.2))
            if trend_ok and volume_ok and (macd_gold or macd_bull) and (kdj_gold or kdj_strong):
                buy_qty = self._buy_qty(code, curr_close)
                if buy_qty <= 0:
                    return None
                return {
                    "strategy_id": self.id,
                    "code": code,
                    "dt": kline["dt"],
                    "direction": "BUY",
                    "price": curr_close,
                    "qty": buy_qty,
                    "stop_loss": curr_close * (1 - stop_loss_pct),
                    "take_profit": curr_close * (1 + take_profit_pct),
                }
            return None

        self._restore_entry_state(code, df, curr_close)
        self.highest_close[code] = max(float(self.highest_close[code]), curr_close)
        entry_price = float(self.entry_price_local[code])
        macd_dead = dif_now < dea_now and dif_prev >= dea_prev
        kdj_dead = k_now < d_now and k_prev >= d_prev
        if kdj_dead and bool(self._cfg("kdj_exit_filter", True)):
            # 主升浪（站上MA20且DIF在零轴上方）中，只认高位死叉；低/中位随机死叉忽略
            strong_trend = trend_ok and dif_now > 0
            cross_level = max(k_prev, d_prev)
            kdj_dead = (not strong_trend) or cross_level >= float(self._cfg("kdj_exit_min_level", 80))
        stop_loss_hit = curr_close <= entry_price * (1 - stop_loss_pct)
        take_profit_hit = curr_close >= entry_price * (1 + take_profit_pct)
        trailing_hit = curr_close <= float(self.highest_close[code]) * (1 - trailing_stop_pct)
        timeout_exit = self.check_max_holding_time(code, max_hold_bars)
        if macd_dead or kdj_dead or stop_loss_hit or take_profit_hit or trailing_hit or timeout_exit:
            reason = []
            if macd_dead:
                reason.append("MACD Death Cross")
            if kdj_dead:
                reason.append("KDJ Death Cross")
            if stop_loss_hit:
                reason.append("Stop Loss")
            if take_profit_hit:
                reason.append("Take Profit")
            if trailing_hit:
                reason.append("Trailing Stop")
            if timeout_exit:
                reason.append("Time Exit")
            return self.create_exit_signal(kline, qty, " | ".join(reason))
        return None


class Strategy12(Strategy11):
    """MACD + KDJ + 成交量共振策略的30分钟版本。"""

    def __init__(self):
        BaseImplementedStrategy.__init__(self, "12", "MACD-KDJ-成交量共振30分钟", trigger_timeframe="30min")
        self._init_state()
