"""A股场内基金（ETF/LOF/REITs）免印花税和过户费，股票照常收取。"""

import pytest

from src.ministries.hu_bu_revenue import HuBuRevenue
from src.utils.market_rules import is_cn_fund


@pytest.mark.parametrize("code, expected", [
    ("510300.SH", True), ("sh510300", True), ("588000.SH", True), ("508000.SH", True),
    ("159915.SZ", True), ("sz159915", True), ("161725.SZ", True), ("180101.SZ", True),
    ("600000.SH", False), ("688981.SH", False), ("000001.SZ", False), ("300750.SZ", False),
    ("830799.BJ", False), ("113050.SH", False), ("AAPL", False), ("", False), (None, False),
])
def test_is_cn_fund(code, expected):
    assert is_cn_fund(code) is expected


def test_funds_exempt_from_stamp_duty_but_stocks_pay():
    hu_bu = HuBuRevenue(1_000_000)
    for fund in ("510300.SH", "159915.SZ"):
        total, commission, stamp, transfer = hu_bu.calculate_cost(100_000, "SELL", 4.0, 25_000, code=fund)
        assert stamp == 0.0 and transfer == 0.0 and total == commission, fund
    _, _, stock_stamp, _ = hu_bu.calculate_cost(100_000, "SELL", 10.0, 10_000, code="600000.SH")
    assert stock_stamp > 0
