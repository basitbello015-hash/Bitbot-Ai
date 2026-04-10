"""
strategies/momentum.py
Momentum Strategy — Priority 3
RSI(14) crosses above/below threshold. BUY on upward cross, SELL on downward cross.
"""

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)
STRATEGY_NAME = "Momentum"


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def generate_signal(
    symbol: str, df: pd.DataFrame,
    rsi_period: int = 14,
    rsi_buy_threshold: float = 52.0,
    rsi_sell_threshold: float = 48.0,
) -> Optional[Dict]:
    if df is None or len(df) < rsi_period + 2:
        return None

    rsi      = _compute_rsi(df["close"], rsi_period)
    rsi_now  = float(rsi.iloc[-1])
    rsi_prev = float(rsi.iloc[-2])

    if rsi_now != rsi_now or rsi_prev != rsi_prev:
        return None

    if rsi_prev < rsi_buy_threshold <= rsi_now:
        return {
            "coin": symbol, "signal": "BUY", "strategy": STRATEGY_NAME,
            "strength": round((rsi_now - 50) / 50, 4),
            "meta": {"rsi": round(rsi_now, 2), "threshold": rsi_buy_threshold},
        }

    if rsi_prev > rsi_sell_threshold >= rsi_now:
        return {
            "coin": symbol, "signal": "SELL", "strategy": STRATEGY_NAME,
            "strength": round((50 - rsi_now) / 50, 4),
            "meta": {"rsi": round(rsi_now, 2), "threshold": rsi_sell_threshold},
        }

    return None
