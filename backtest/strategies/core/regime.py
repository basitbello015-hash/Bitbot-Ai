"""
core/regime.py
Market Regime Detection Module.

Classifies each coin as TRENDING, RANGING, or UNCLEAR before any strategy
is evaluated. Only TRENDING coins proceed to the normal strategy stack.
RANGING coins activate Grid Mode. UNCLEAR coins are skipped entirely.

Indicators used:
  - ADX(14)         — trend strength
  - MA(20), MA(50)  — direction and separation
  - Recent N-bar high/low vs close — price structure vs MAs

Classification Rules:
  TRENDING  : ADX > 25  AND  MA20/MA50 separation ≥ threshold
              AND close consistently above or below MA50
  RANGING   : ADX < 20  AND  MA20/MA50 flat or overlapping
              AND price oscillates within recent high/low range
  UNCLEAR   : Neither condition met → skip the coin
"""

import logging
from enum import Enum
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING = "TRENDING"
    RANGING  = "RANGING"
    UNCLEAR  = "UNCLEAR"


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    tr = (
        (high - low)
        .combine((high - close.shift(1)).abs(), max)
        .combine((low  - close.shift(1)).abs(), max)
    )

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr      = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, float("nan"))
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, float("nan"))

    dx_denom = (plus_di + minus_di).replace(0, float("nan"))
    dx  = 100 * (plus_di - minus_di).abs() / dx_denom
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def _ma_separation_pct(ma_fast: float, ma_slow: float) -> float:
    if ma_slow == 0:
        return 0.0
    return abs(ma_fast - ma_slow) / ma_slow * 100


def _is_ma_flat(series: pd.Series, lookback: int = 5, threshold_pct: float = 0.5) -> bool:
    if len(series) < lookback + 1:
        return False
    oldest = float(series.iloc[-(lookback + 1)])
    newest = float(series.iloc[-1])
    if oldest == 0:
        return False
    return abs(newest - oldest) / oldest * 100 < threshold_pct


def _close_consistently_above_ma(close: pd.Series, ma: pd.Series, lookback: int = 5) -> bool:
    return bool((close.iloc[-lookback:] > ma.iloc[-lookback:]).all())


def _close_consistently_below_ma(close: pd.Series, ma: pd.Series, lookback: int = 5) -> bool:
    return bool((close.iloc[-lookback:] < ma.iloc[-lookback:]).all())


def _price_oscillates_in_range(
    close: pd.Series, high: pd.Series, low: pd.Series,
    lookback: int = 20, containment_pct: float = 0.80,
) -> bool:
    recent_high = float(high.iloc[-lookback:].max())
    recent_low  = float(low.iloc[-lookback:].min())
    if recent_high == recent_low:
        return False
    recent_closes = close.iloc[-lookback:]
    inside = ((recent_closes >= recent_low) & (recent_closes <= recent_high)).sum()
    return inside / lookback >= containment_pct


def detect_regime(
    df: pd.DataFrame,
    adx_period:         int   = 14,
    adx_trend_thresh:   float = 25.0,
    adx_range_thresh:   float = 20.0,
    ma_fast_period:     int   = 20,
    ma_slow_period:     int   = 50,
    ma_sep_thresh_pct:  float = 0.5,
    ma_flat_lookback:   int   = 5,
    ma_flat_thresh_pct: float = 0.5,
    price_lookback:     int   = 5,
    range_lookback:     int   = 20,
) -> Tuple[Regime, dict]:
    required = ma_slow_period + max(adx_period, range_lookback) + 2
    if df is None or len(df) < required:
        return Regime.UNCLEAR, {"reason": "insufficient_data"}

    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    adx     = _compute_adx(df, adx_period)
    ma_fast = close.rolling(ma_fast_period).mean()
    ma_slow = close.rolling(ma_slow_period).mean()

    adx_now     = float(adx.iloc[-1])
    ma_fast_now = float(ma_fast.iloc[-1])
    ma_slow_now = float(ma_slow.iloc[-1])
    close_now   = float(close.iloc[-1])
    sep_pct     = _ma_separation_pct(ma_fast_now, ma_slow_now)

    trend_adx   = adx_now >= adx_trend_thresh
    trend_sep   = sep_pct >= ma_sep_thresh_pct
    trend_price = (
        _close_consistently_above_ma(close, ma_slow, price_lookback)
        or _close_consistently_below_ma(close, ma_slow, price_lookback)
    )

    range_adx  = adx_now <= adx_range_thresh
    range_ma   = (
        _is_ma_flat(ma_fast, ma_flat_lookback, ma_flat_thresh_pct)
        and _is_ma_flat(ma_slow, ma_flat_lookback, ma_flat_thresh_pct)
    )
    range_price = _price_oscillates_in_range(close, high, low, range_lookback)

    metrics = {
        "adx": round(adx_now, 2), "ma_fast": round(ma_fast_now, 6),
        "ma_slow": round(ma_slow_now, 6), "ma_sep_pct": round(sep_pct, 4),
        "close": round(close_now, 6),
        "trend_adx": trend_adx, "trend_sep": trend_sep, "trend_price": trend_price,
        "range_adx": range_adx, "range_ma": range_ma, "range_price": range_price,
    }

    trending_score = sum([trend_adx, trend_sep, trend_price])
    ranging_score  = sum([range_adx, range_ma, range_price])

    if trending_score >= 2 and ranging_score < 2:
        regime = Regime.TRENDING
    elif ranging_score >= 2 and trending_score < 2:
        regime = Regime.RANGING
    else:
        regime = Regime.UNCLEAR

    metrics["trending_score"] = trending_score
    metrics["ranging_score"]  = ranging_score
    return regime, metrics
