"""
strategies/trend.py
Trend Following Strategy — Priority 2
MA(ma_fast) / MA(ma_slow) crossover. BUY on golden cross, SELL on death cross.
"""

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)
STRATEGY_NAME = "TrendFollowing"


def generate_signal(
    symbol: str, df: pd.DataFrame, ma_fast: int = 10, ma_slow: int = 50,
) -> Optional[Dict]:
    required_bars = ma_slow + 2
    if df is None or len(df) < required_bars:
        return None

    close   = df["close"]
    fast_ma = close.rolling(ma_fast).mean()
    slow_ma = close.rolling(ma_slow).mean()

    fast_now, fast_prev = float(fast_ma.iloc[-1]), float(fast_ma.iloc[-2])
    slow_now, slow_prev = float(slow_ma.iloc[-1]), float(slow_ma.iloc[-2])

    if any(v != v for v in [fast_now, fast_prev, slow_now, slow_prev]):
        return None

    if fast_prev <= slow_prev and fast_now > slow_now:
        return {
            "coin": symbol, "signal": "BUY", "strategy": STRATEGY_NAME,
            "strength": round((fast_now - slow_now) / slow_now, 6),
            "meta": {f"ma{ma_fast}": round(fast_now, 6), f"ma{ma_slow}": round(slow_now, 6), "cross": "golden"},
        }

    if fast_prev >= slow_prev and fast_now < slow_now:
        return {
            "coin": symbol, "signal": "SELL", "strategy": STRATEGY_NAME,
            "strength": round((slow_now - fast_now) / slow_now, 6),
            "meta": {f"ma{ma_fast}": round(fast_now, 6), f"ma{ma_slow}": round(slow_now, 6), "cross": "death"},
        }

    return None
