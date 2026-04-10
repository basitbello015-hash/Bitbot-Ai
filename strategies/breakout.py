"""
strategies/breakout.py
Breakout Strategy — Priority 1 (HIGHEST)

Improvements over naive breakout:
  - SL/TP embedded in signal output
  - Volume confirmation (breakout bar must exceed avg × volume_factor)
  - Consolidation check (price near level for ≥ consolidation_bars)
  - 2-bar close confirmation (rejects single-bar spikes)
"""

import logging
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)
STRATEGY_NAME = "Breakout"
_CONSOLIDATION_PROXIMITY_PCT = 0.015


def _is_consolidating_near(window: pd.DataFrame, level: float, consolidation_bars: int) -> bool:
    proximity = level * _CONSOLIDATION_PROXIMITY_PCT
    near = ((window["close"] - level).abs() <= proximity).sum()
    return int(near) >= consolidation_bars


def generate_signal(
    symbol: str,
    df: pd.DataFrame,
    lookback_bars: int = 20,
    breakout_buffer_pct: float = 0.002,
    volume_factor: float = 1.5,
    consolidation_bars: int = 3,
    min_rr: float = 2.0,
) -> Optional[Dict]:
    required_bars = lookback_bars + 2
    if df is None or len(df) < required_bars:
        return None

    window        = df.iloc[-(required_bars):-2]
    prev_close    = float(df["close"].iloc[-2])
    current_close = float(df["close"].iloc[-1])
    current_vol   = float(df["volume"].iloc[-1])
    avg_vol       = float(window["volume"].mean())

    resistance = float(window["high"].max())
    support    = float(window["low"].min())

    breakout_up_level   = resistance * (1 + breakout_buffer_pct)
    breakout_down_level = support    * (1 - breakout_buffer_pct)

    volume_confirmed = (avg_vol > 0) and (current_vol >= avg_vol * volume_factor)

    # BUY signal
    if (
        current_close > breakout_up_level
        and prev_close > resistance
        and volume_confirmed
        and _is_consolidating_near(window, resistance, consolidation_bars)
    ):
        sl_price = resistance * (1 - breakout_buffer_pct)
        sl_dist  = abs(current_close - sl_price)
        return {
            "coin": symbol, "signal": "BUY", "strategy": STRATEGY_NAME,
            "strength": round((current_close - resistance) / resistance * (current_vol / avg_vol), 6),
            "stop_loss":   round(sl_price, 8),
            "take_profit": round(current_close + sl_dist * min_rr, 8),
            "meta": {
                "resistance": round(resistance, 6),
                "breakout_level": round(breakout_up_level, 6),
                "close": current_close,
                "volume_ratio": round(current_vol / avg_vol, 2),
            },
        }

    # SELL signal
    if (
        current_close < breakout_down_level
        and prev_close < support
        and volume_confirmed
        and _is_consolidating_near(window, support, consolidation_bars)
    ):
        sl_price = support * (1 + breakout_buffer_pct)
        sl_dist  = abs(sl_price - current_close)
        return {
            "coin": symbol, "signal": "SELL", "strategy": STRATEGY_NAME,
            "strength": round((support - current_close) / support * (current_vol / avg_vol), 6),
            "stop_loss":   round(sl_price, 8),
            "take_profit": round(current_close - sl_dist * min_rr, 8),
            "meta": {
                "support": round(support, 6),
                "breakout_level": round(breakout_down_level, 6),
                "close": current_close,
                "volume_ratio": round(current_vol / avg_vol, 2),
            },
        }

    return None
