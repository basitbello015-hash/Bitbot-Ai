"""
strategies/volume.py
Volume Spike Strategy — Priority 4
Volume ≥ avg × multiplier + bullish/bearish candle body direction.
"""

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)
STRATEGY_NAME = "VolumeSpike"


def generate_signal(
    symbol: str, df: pd.DataFrame,
    volume_ma_period: int = 20,
    volume_multiplier: float = 2.0,
) -> Optional[Dict]:
    required_bars = volume_ma_period + 1
    if df is None or len(df) < required_bars:
        return None

    volume_history = df["volume"].iloc[-(required_bars):-1]
    volume_avg     = float(volume_history.mean())
    current_volume = float(df["volume"].iloc[-1])
    current_open   = float(df["open"].iloc[-1])
    current_close  = float(df["close"].iloc[-1])

    if volume_avg == 0:
        return None

    ratio = current_volume / volume_avg
    if ratio < volume_multiplier:
        return None

    if current_close > current_open:
        signal_dir = "BUY"
    elif current_close < current_open:
        signal_dir = "SELL"
    else:
        return None

    return {
        "coin": symbol, "signal": signal_dir, "strategy": STRATEGY_NAME,
        "strength": round((ratio - volume_multiplier) / volume_multiplier, 4),
        "meta": {"volume": round(current_volume, 2), "volume_avg": round(volume_avg, 2), "volume_ratio": round(ratio, 2)},
    }
