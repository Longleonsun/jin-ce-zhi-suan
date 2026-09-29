# src/ministries/hu_bu_revenue.py
import pandas as pd
from src.utils.constants import *
from src.utils.runtime_params import get_value
from src.utils.market_rules import is_cn_fund, is_us_symbol, market_profile

class HuBuRevenue:
    """
    户部 (Revenue): 逐笔核算资金、净值、手续费、印花税、总成本
    """
    def __init__(self, initial_capital):
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.transactions = []
        self.daily_nav = [] # Date, NAV, Cash, Holdings Value
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_transfer_fee = 0.0

    def calculate_cost(self, amount, direction, price, quantity, code=None):
        """
        Calculate transaction cost.
        Returns: (total_cost, commission, stamp_duty, transfer_fee)
        """
        # 按标的市场取费率：A股读 trading_cost，美股读 trading_cost_us
        if code is not None and is_us_symbol(code):
            profile = market_profile(code)
            key, defaults = profile["cost_config_key"], profile["cost_defaults"]
        else:
            key = "trading_cost"
            defaults = {"commission_rate": COMMISSION_RATE, "min_commission": MIN_COMMISSION, "stamp_duty": STAMP_DUTY, "transfer_fee": TRANSFER_FEE}
        # Commission: Max(min_commission, amount * commission_rate)
        min_commission = float(get_value(f"{key}.min_commission", defaults["min_commission"]))
        commission_rate = float(get_value(f"{key}.commission_rate", defaults["commission_rate"]))
        stamp_duty_rate = float(get_value(f"{key}.stamp_duty", defaults["stamp_duty"]))
        transfer_fee_rate = float(get_value(f"{key}.transfer_fee", defaults["transfer_fee"]))
        commission = max(min_commission, amount * commission_rate)
        # 印花税、过户费只对股票征收，场内基金免收
        if is_cn_fund(code):
            stamp_duty_rate = transfer_fee_rate = 0.0
        
        # Stamp Duty: 0.1% on SELL only
        stamp_duty = 0.0
        if direction == 'SELL':
            stamp_duty = amount * stamp_duty_rate
            
        # Transfer Fee: amount * 0.00001
        transfer_fee = amount * transfer_fee_rate
        
        total_cost = commission + stamp_duty + transfer_fee
        return total_cost, commission, stamp_duty, transfer_fee

    def record_transaction(self, strategy_id, dt, direction, price, quantity, cost, pnl=0.0, commission=0.0, stamp_duty=0.0, transfer_fee=0.0, reason=""):
        """
        Record a transaction.
        """
        amount = price * quantity
        
        self.transactions.append({
            'strategy_id': strategy_id,
            'dt': dt,
            'direction': direction,
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'cost': cost,
            'pnl': pnl,
            'commission': float(commission or 0.0),
            'stamp_duty': float(stamp_duty or 0.0),
            'transfer_fee': float(transfer_fee or 0.0),
            'reason': str(reason or '')  # 系统操作为 FORCE_CLOSE_END/DRAWDOWN_LIMIT/STOP_LOSS/TAKE_PROFIT，策略信号为策略自带说明
        })
        
        if direction == 'BUY':
            self.cash -= (amount + cost)
        elif direction == 'SELL':
            self.cash += (amount - cost) # Proceeds - Cost
            
        self.total_commission += float(commission or 0.0)
        self.total_stamp_duty += float(stamp_duty or 0.0)
        self.total_transfer_fee += float(transfer_fee or 0.0)
        
    def update_daily_nav(self, dt, holdings_value):
        nav = self.cash + holdings_value
        self.daily_nav.append({
            'dt': dt,
            'nav': nav,
            'cash': self.cash,
            'holdings': holdings_value
        })
        return nav

    def get_nav_history(self):
        return pd.DataFrame(self.daily_nav)
