"""
strategies/mean_reversion.py
Mean Reversion Strategy — Priority 5 (LOWEST)
Bollinger Bands (20,2) extremes + RSI confirmation. Both conditions must hold.
"""

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)
STRATEGY_NAME = "MeanReversion"


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
    bb_period: int = 20, bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0, rsi_overbought: float = 70.0,
) -> Optional[Dict]:
    if df is None or len(df) < max(bb_period, rsi_period) + 2:
        return None

    close = df["close"]
    bb_sma        = close.rolling(bb_period).mean()
    bb_std_series = close.rolling(bb_period).std()
    upper_band    = bb_sma + bb_std * bb_std_series
    lower_band    = bb_sma - bb_std * bb_std_series

    current_close = float(close.iloc[-1])
    upper_now     = float(upper_band.iloc[-1])
    lower_now     = float(lower_band.iloc[-1])
    rsi_now       = float(_compute_rsi(close, rsi_period).iloc[-1])

    if any(v != v for v in [upper_now, lower_now, rsi_now]):
        return None

    band_width = upper_now - lower_now

    if current_close <= lower_now and rsi_now <= rsi_oversold:
        dist = (lower_now - current_close) / band_width if band_width > 0 else 0
        return {
            "coin": symbol, "signal": "BUY", "strategy": STRATEGY_NAME,
            "strength": round(dist + (rsi_oversold - rsi_now) / 100, 4),
            "meta": {"close": current_close, "bb_lower": round(lower_now, 6), "bb_upper": round(upper_now, 6), "rsi": round(rsi_now, 2)},
        }

    if current_close >= upper_now and rsi_now >= rsi_overbought:
        dist = (current_close - upper_now) / band_width if band_width > 0 else 0
        return {
            "coin": symbol, "signal": "SELL", "strategy": STRATEGY_NAME,
            "strength": round(dist + (rsi_now - rsi_overbought) / 100, 4),
            "meta": {"close": current_close, "bb_lower": round(lower_now, 6), "bb_upper": round(upper_now, 6), "rsi": round(rsi_now, 2)},
        }

    return None
