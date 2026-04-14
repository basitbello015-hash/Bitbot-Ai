import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Literal, List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)
STRATEGY_NAME = "TrendFollowing"


# ----------------------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------------------

class TrendSignal(BaseModel):
    triggered: bool
    entry_type: Optional[Literal["pullback", "continuation", "ema_reclaim"]] = None
    confidence: float          # 0–1
    metadata: dict

# ----------------------------------------------------------------------
# Helper: Technical Indicators
# ----------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def _linreg_slope(series: pd.Series, lookback: int) -> float:
    """Slope of linear regression over last `lookback` values."""
    if len(series) < lookback:
        return 0.0
    y = series.iloc[-lookback:].values.astype(float)
    x = np.arange(lookback)
    slope = np.polyfit(x, y, 1)[0]
    return slope

def _efficiency_ratio(close: pd.Series, lookback: int) -> float:
    """Efficiency ratio = abs(change) / sum of absolute differences."""
    if len(close) < lookback:
        return 0.0
    change = close.iloc[-1] - close.iloc[-lookback]
    volatility = close.diff().abs().iloc[-lookback+1:].sum()
    if volatility == 0:
        return 0.0
    return min(abs(change) / volatility, 1.0)

# ----------------------------------------------------------------------
# Swing High / Low Detection
# ----------------------------------------------------------------------

def _swing_points(
    df: pd.DataFrame,
    column: str,
    order: int,
    tolerance: float,
    mode: Literal['high', 'low']
) -> List[Tuple[int, float]]:
    """
    Detect swing highs or lows.
    Returns list of (index, price) tuples.
    """
    points = []
    n = len(df)
    for i in range(order, n - order):
        window_left = df[column].iloc[i - order:i].values
        window_right = df[column].iloc[i + 1:i + order + 1].values
        min_neighbor = min(np.min(window_left), np.min(window_right))
        max_neighbor = max(np.max(window_left), np.max(window_right))
        current = df[column].iloc[i]

        if mode == 'high':
            # Swing high: current >= max neighbor * (1 - tolerance)
            if current >= max_neighbor * (1 - tolerance):
                points.append((i, current))
        else:  # low
            # Swing low: current <= min neighbor * (1 + tolerance)
            if current <= min_neighbor * (1 + tolerance):
                points.append((i, current))
    return points

# ----------------------------------------------------------------------
# Trend Structure Detection
# ----------------------------------------------------------------------

def _detect_trend_structure(df: pd.DataFrame, config: dict) -> dict:
    """
    Detects uptrend, computes strength score and swing points.
    Returns dict with keys: is_uptrend, trend_strength, trend_state,
    swing_highs, swing_lows, rejection_reason.
    """
    # Required bars: need at least swing_order*2 + 1 + extra for other indicators
    min_bars = config.get('swing_order', 3) * 2 + 1 + max(
        config.get('er_lookback', 50),
        config.get('slope_lookback', 10),
        config.get('volume_lookback', 20)
    )
    if len(df) < min_bars:
        raise ValueError(f"Insufficient data: need at least {min_bars} bars")

    swing_order = config.get('swing_order', 3)
    swing_tolerance = config.get('swing_tolerance', 0.001)  # 0.1%
    min_higher_highs = config.get('min_higher_highs', 2)
    min_higher_lows = config.get('min_higher_lows', 1)
    structure_break_tolerance = config.get('structure_break_tolerance', 0.01)

    # Get swing highs and lows
    swing_highs = _swing_points(df, 'high', swing_order, swing_tolerance, 'high')
    swing_lows = _swing_points(df, 'low', swing_order, swing_tolerance, 'low')

    # Default return
    result = {
        'is_uptrend': False,
        'trend_strength': 0.0,
        'trend_state': 'weak',
        'swing_highs': [p for _, p in swing_highs],
        'swing_lows': [p for _, p in swing_lows],
        'rejection_reason': None
    }

    # Need at least some swings
    if len(swing_highs) < min_higher_highs or len(swing_lows) < 1:
        result['rejection_reason'] = 'insufficient_swings'
        return result

    # Extract prices and indices
    high_prices = [p for _, p in swing_highs]
    low_prices = [p for _, p in swing_lows]
    high_indices = [i for i, _ in swing_highs]
    low_indices = [i for i, _ in swing_lows]

    # Check for consecutive higher highs
    higher_highs = 0
    for j in range(1, len(high_prices)):
        if high_prices[j] > high_prices[j-1]:
            higher_highs += 1
        else:
            higher_highs = 0  # reset if not higher
        if higher_highs >= min_higher_highs - 1:
            break
    if higher_highs < min_higher_highs - 1:
        result['rejection_reason'] = 'not_enough_higher_highs'
        return result

    # Find the index of the first higher high that starts the sequence
    # We need at least one higher low after that first HH
    # Simple: find the most recent higher high sequence start
    # Get the index of the first high in the consecutive sequence
    seq_start_idx = None
    for k in range(1, len(high_prices)):
        if high_prices[k] > high_prices[k-1]:
            if seq_start_idx is None:
                seq_start_idx = k-1
            if k - seq_start_idx + 1 >= min_higher_highs:
                # sequence found, the first HH index is seq_start_idx
                first_hh_index = high_indices[seq_start_idx]
                break
        else:
            seq_start_idx = None
    else:
        result['rejection_reason'] = 'no_valid_higher_high_sequence'
        return result

    # Check for higher lows after first HH
    higher_lows_after = 0
    for low_idx, low_price in zip(low_indices, low_prices):
        if low_idx > first_hh_index:
            # Find previous higher low (the last higher low before this one)
            # For simplicity, we just need at least min_higher_lows that are higher than the previous higher low
            # Actually spec: "higher swing low after the first HH" -> any low that is higher than the previous low
            # We'll count consecutive higher lows after first HH
            # We need to iterate over lows after first_hh_index in order
    # Better approach: collect lows after first HH in order
    lows_after = [(idx, price) for idx, price in swing_lows if idx > first_hh_index]
    if len(lows_after) < min_higher_lows:
        result['rejection_reason'] = 'not_enough_higher_lows_after_HH'
        return result
    higher_low_count = 1
    for i in range(1, len(lows_after)):
        if lows_after[i][1] > lows_after[i-1][1]:
            higher_low_count += 1
        else:
            higher_low_count = 1
        if higher_low_count >= min_higher_lows:
            break
    if higher_low_count < min_higher_lows:
        result['rejection_reason'] = 'lows_not_consecutively_higher'
        return result

    # Check no lower low below most recent higher low (with tolerance)
    if lows_after:
        most_recent_hl = lows_after[-1][1]
        # Look at all swing lows after first HH, ensure none is below most_recent_hl * (1 - tolerance)
        for _, low_price in lows_after:
            if low_price < most_recent_hl * (1 - structure_break_tolerance):
                result['rejection_reason'] = 'lower_low_break'
                return result

    # Market structure passed -> score 1.0 for this component
    structure_score = 1.0

    # Efficiency Ratio
    er_lookback = config.get('er_lookback', 50)
    er_threshold = config.get('er_threshold', 0.3)
    er = _efficiency_ratio(df['close'], er_lookback)
    er_score = min(er / er_threshold, 1.0)

    # Moving Average Alignment
    ema_fast = config.get('ema_fast', 20)
    ema_slow = config.get('ema_slow', 50)
    ema_tolerance = config.get('ema_tolerance', 0.005)
    ema_fast_vals = _ema(df['close'], ema_fast)
    ema_slow_vals = _ema(df['close'], ema_slow)
    fast_above_slow = ema_fast_vals.iloc[-1] >= ema_slow_vals.iloc[-1]
    price_above_fast = df['close'].iloc[-1] >= ema_fast_vals.iloc[-1] * (1 - ema_tolerance)
    if fast_above_slow and price_above_fast:
        ma_score = 1.0
    elif fast_above_slow:
        ma_score = 0.5
    else:
        ma_score = 0.0

    # EMA Slope (slow EMA)
    slope_lookback = config.get('slope_lookback', 10)
    slope_target = config.get('slope_target', 0.001)
    slow_ema_slope = _linreg_slope(ema_slow_vals, slope_lookback)
    slope_score = np.clip(slow_ema_slope / slope_target, 0.0, 1.0)

    # Volume Trend
    volume_lookback = config.get('volume_lookback', 20)
    volume_slope = _linreg_slope(df['volume'], volume_lookback)
    # slope of volume: positive adds confidence, score capped at 1
    volume_score = np.clip(volume_slope / 0.01, 0.0, 1.0)

    # Weighted composite score
    w_struct = config.get('weight_structure', 0.30)
    w_er = config.get('weight_er', 0.25)
    w_ma = config.get('weight_ma', 0.20)
    w_slope = config.get('weight_slope', 0.15)
    w_vol = config.get('weight_volume', 0.10)

    trend_strength = (structure_score * w_struct +
                      er_score * w_er +
                      ma_score * w_ma +
                      slope_score * w_slope +
                      volume_score * w_vol)

    # State
    if trend_strength < 0.35:
        state = 'weak'
    elif trend_strength < 0.65:
        state = 'moderate'
    else:
        state = 'strong'

    result.update({
        'is_uptrend': True,
        'trend_strength': trend_strength,
        'trend_state': state,
        'rejection_reason': None
    })
    return result

# ----------------------------------------------------------------------
# Volatility Filter (Hard Reject)
# ----------------------------------------------------------------------

def _volatility_reject(df: pd.DataFrame, config: dict) -> bool:
    """Returns True if volatility exceeds threshold (hard reject)."""
    atr_period = 14
    lookback = 20
    max_multiple = config.get('max_atr_multiple', 3.0)

    if len(df) < lookback + atr_period:
        return False  # insufficient data, allow signal

    atr_vals = _atr(df, atr_period)
    # median ATR over last `lookback` bars
    recent_atr = atr_vals.iloc[-lookback:]
    median_atr = recent_atr.median()
    current_atr = atr_vals.iloc[-1]
    if current_atr > median_atr * max_multiple:
        return True
    return False

# ----------------------------------------------------------------------
# Entry Conditions
# ----------------------------------------------------------------------

def _pullback_entry(
    df: pd.DataFrame,
    trend_struct: dict,
    config: dict
) -> Tuple[bool, float, dict]:
    """Pullback entry detection."""
    if not trend_struct['is_uptrend']:
        return False, 0.0, {}
    min_strength = config.get('min_pullback_trend_strength', 0.4)
    if trend_struct['trend_strength'] < min_strength:
        return False, 0.0, {}

    impulse_lookback = config.get('impulse_lookback', 20)
    if len(df) < impulse_lookback:
        return False, 0.0, {}

    # Recent high and low before that high
    recent_high = df['high'].iloc[-impulse_lookback:].max()
    recent_high_idx = df['high'].iloc[-impulse_lookback:].idxmax()
    # Find lowest low before that high within the same window
    window = df.loc[df.index >= df.index[-impulse_lookback], :]
    prior_lows = window.loc[:recent_high_idx, 'low']
    low_before_impulse = prior_lows.min() if not prior_lows.empty else df['low'].iloc[-impulse_lookback]

    current_low = df['low'].iloc[-1]
    # Avoid division by zero
    if recent_high == low_before_impulse:
        depth = 0.0
    else:
        depth = (recent_high - current_low) / (recent_high - low_before_impulse)
    depth_min = config.get('pullback_depth_min', 0.15)
    depth_max = config.get('pullback_depth_max', 0.60)
    if not (depth_min <= depth <= depth_max):
        return False, 0.0, {}

    # EMA20 proximity
    ema20 = _ema(df['close'], 20)
    ema_proximity = config.get('ema_proximity', 0.01)
    if abs(df['close'].iloc[-1] - ema20.iloc[-1]) / df['close'].iloc[-1] > ema_proximity:
        return False, 0.0, {}

    # Confirmation candlestick
    confirm_score = 0.0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last['close'] - last['open'])
    body_pct = body / last['close']
    # Bullish hammer
    lower_wick = min(last['close'], last['open']) - last['low']
    if lower_wick > 2 * body and body_pct < 0.003:
        confirm_score = max(confirm_score, 0.5)
    # Bullish engulfing
    if last['close'] > prev['high'] and last['open'] < prev['close']:
        confirm_score = max(confirm_score, 0.5)
    # Inside bar breakout
    prev_range = (prev['high'] - prev['low']) / prev['close']
    if prev_range < 0.005 and last['high'] > prev['high']:
        confirm_score = max(confirm_score, 0.5)

    # If two conditions met, higher score
    # We'll check how many distinct conditions are true (simplified)
    conditions = 0
    if lower_wick > 2 * body and body_pct < 0.003:
        conditions += 1
    if last['close'] > prev['high'] and last['open'] < prev['close']:
        conditions += 1
    if prev_range < 0.005 and last['high'] > prev['high']:
        conditions += 1
    if conditions >= 2:
        confirm_score = 1.0

    if confirm_score == 0.0:
        return False, 0.0, {}

    # Confidence
    depth_factor = 1 - (depth / depth_max)  # shallower pullback gives higher factor
    confidence = depth_factor * 0.6 + confirm_score * 0.4
    confidence = np.clip(confidence, 0.0, 1.0)

    metadata = {
        'pullback_depth': depth,
        'confirmation_score': confirm_score,
        'recent_high': recent_high,
        'low_before_impulse': low_before_impulse
    }
    return True, confidence, metadata

def _continuation_entry(
    df: pd.DataFrame,
    trend_struct: dict,
    config: dict
) -> Tuple[bool, float, dict]:
    """Flag/pennant continuation entry."""
    if not trend_struct['is_uptrend']:
        return False, 0.0, {}
    min_strength = config.get('min_continuation_trend_strength', 0.5)
    if trend_struct['trend_strength'] < min_strength:
        return False, 0.0, {}

    min_bars = config.get('consolidation_min_bars', 3)
    max_bars = config.get('consolidation_max_bars', 15)
    max_range = config.get('max_consolidation_range', 0.04)
    breakout_tol = config.get('breakout_tolerance', 0.001)

    # Need enough bars to look back
    if len(df) < max_bars + 1:
        return False, 0.0, {}

    # Scan for consolidation of length between min_bars and max_bars ending at current bar
    for length in range(min_bars, min(max_bars, len(df)-1) + 1):
        segment = df.iloc[-length-1:-1]  # exclude current bar for range detection
        high_seg = segment['high'].max()
        low_seg = segment['low'].min()
        range_pct = (high_seg - low_seg) / low_seg
        if range_pct <= max_range:
            # Check breakout: current close > consolidation high + tolerance
            breakout_price = high_seg * (1 + breakout_tol)
            if df['close'].iloc[-1] > breakout_price:
                confidence = trend_struct['trend_strength'] * 0.7 + (1 - range_pct / max_range) * 0.3
                confidence = np.clip(confidence, 0.0, 1.0)
                metadata = {
                    'consolidation_length': length,
                    'consolidation_range': range_pct,
                    'breakout_price': breakout_price
                }
                return True, confidence, metadata
    return False, 0.0, {}

def _ema_reclaim_entry(
    df: pd.DataFrame,
    trend_struct: dict,
    config: dict
) -> Tuple[bool, float, dict]:
    """EMA reclaim entry after brief dip below EMA20."""
    if not trend_struct['is_uptrend']:
        return False, 0.0, {}

    min_below = config.get('ema_reclaim_min_bars_below', 2)
    max_below = config.get('ema_reclaim_max_bars_below', 8)

    ema20 = _ema(df['close'], 20)
    if len(ema20) < max_below + 2:
        return False, 0.0, {}

    # Price below EMA20 for at least min_below and at most max_below bars
    below = df['close'] < ema20
    # Count consecutive below bars ending at index -2 (since we need cross at last bar)
    # We need current bar (last) to cross above
    if not (df['close'].iloc[-1] > ema20.iloc[-1] and df['close'].iloc[-2] <= ema20.iloc[-2]):
        return False, 0.0, {}

    # Count how many consecutive below before the cross
    below_count = 0
    for i in range(-2, -max_below-2, -1):
        if below.iloc[i]:
            below_count += 1
        else:
            break
    if not (min_below <= below_count <= max_below):
        return False, 0.0, {}

    # Optional follow-through (next bar not available in current df, so we cannot use)
    # According to spec: "next bar (if available) shows bullish follow-through" – we don't have next bar.
    # So we ignore the follow-through part or treat as 0.
    follow_through = 0.0

    confidence = trend_struct['trend_strength'] * 0.8 + follow_through * 0.2
    confidence = np.clip(confidence, 0.0, 1.0)
    metadata = {'bars_below': below_count}
    return True, confidence, metadata

# ----------------------------------------------------------------------
# Multi-Timeframe Confirmation
# ----------------------------------------------------------------------

def _mtf_confirm(htf_df: pd.DataFrame, config: dict) -> Tuple[bool, dict]:
    """Check HTF trend confirmation."""
    htf_min = config.get('htf_min_strength', 0.3)
    try:
        htf_struct = _detect_trend_structure(htf_df, config)
    except ValueError:
        return False, {'error': 'insufficient_htf_data'}
    confirmed = (htf_struct['is_uptrend'] and
                 htf_struct['trend_strength'] >= htf_min)
    return confirmed, htf_struct

# ----------------------------------------------------------------------
# Main Detection Function
# ----------------------------------------------------------------------

def detect_trend(
    ltf_df: pd.DataFrame,
    htf_df: pd.DataFrame,
    symbol: str,
    config: dict
) -> TrendSignal:
    """
    Pure function to detect trend following signal.
    Returns TrendSignal with entry type and confidence.
    """
    # Validate input dataframes
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    for df, name in [(ltf_df, 'LTF'), (htf_df, 'HTF')]:
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"{name} DataFrame missing required columns")
        if len(df) < 10:
            raise ValueError(f"{name} DataFrame has insufficient rows")

    # Hard volatility reject on LTF
    if _volatility_reject(ltf_df, config):
        return TrendSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={'rejection_reason': 'excessive_volatility'}
        )

    # Detect LTF trend structure
    try:
        ltf_trend = _detect_trend_structure(ltf_df, config)
    except ValueError as e:
        return TrendSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={'error': str(e)}
        )

    if not ltf_trend['is_uptrend']:
        return TrendSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={'rejection_reason': ltf_trend.get('rejection_reason', 'not_uptrend')}
        )

    # MTF confirmation
    htf_confirmed, htf_meta = _mtf_confirm(htf_df, config)
    if not htf_confirmed:
        return TrendSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={
                'rejection_reason': 'htf_not_confirmed',
                'htf_trend': htf_meta
            }
        )

    # Evaluate entries in priority order: pullback > continuation > ema_reclaim
    # Pullback
    pull_triggered, pull_conf, pull_meta = _pullback_entry(ltf_df, ltf_trend, config)
    if pull_triggered:
        return TrendSignal(
            triggered=True,
            entry_type='pullback',
            confidence=pull_conf,
            metadata={
                'ltf_trend': ltf_trend,
                'htf_trend': htf_meta,
                'entry_details': pull_meta
            }
        )

    # Continuation
    cont_triggered, cont_conf, cont_meta = _continuation_entry(ltf_df, ltf_trend, config)
    if cont_triggered:
        return TrendSignal(
            triggered=True,
            entry_type='continuation',
            confidence=cont_conf,
            metadata={
                'ltf_trend': ltf_trend,
                'htf_trend': htf_meta,
                'entry_details': cont_meta
            }
        )

    # EMA Reclaim
    ema_triggered, ema_conf, ema_meta = _ema_reclaim_entry(ltf_df, ltf_trend, config)
    if ema_triggered:
        return TrendSignal(
            triggered=True,
            entry_type='ema_reclaim',
            confidence=ema_conf,
            metadata={
                'ltf_trend': ltf_trend,
                'htf_trend': htf_meta,
                'entry_details': ema_meta
            }
        )

    # No entry
    return TrendSignal(
        triggered=False,
        entry_type=None,
        confidence=0.0,
        metadata={'rejection_reason': 'no_entry_condition_met'}
    )
