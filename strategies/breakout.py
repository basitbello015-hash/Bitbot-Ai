"""
strategies/breakout_strategy.py

Institutional-grade Breakout Strategy — Plug-in Module.

Provides:
    BreakoutSignal   — Pydantic output model
    detect_breakout  — pure function, no side effects

Entry types supported:
    "breakout"      — direct close above resistance with volume spike
    "retest"        — price returns to resistance after confirmed break
    "continuation"  — shallow pullback after strong break, then new high

Multi-timeframe confirmation required on HTF before any signal is emitted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

try:
    import polars as pl
    _POLARS = True
except ImportError:
    _POLARS = False

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
STRATEGY_NAME = "Breakout"



# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class BreakoutSignal(BaseModel):
    triggered:   bool
    entry_type:  Optional[Literal["breakout", "retest", "continuation"]]
    confidence:  float = Field(ge=0.0, le=1.0)
    metadata:    Dict[str, Any]


# ---------------------------------------------------------------------------
# DataFrame normalisation helpers
# ---------------------------------------------------------------------------

def _to_pandas(df) -> pd.DataFrame:
    """Accept either polars or pandas DataFrame; always return pandas."""
    if _POLARS and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df
    raise ValueError(f"Unsupported DataFrame type: {type(df)}")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a column as float64 Series; raise clearly if missing."""
    if name not in df.columns:
        raise ValueError(f"Required column '{name}' missing from DataFrame.")
    return df[name].astype(float)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder smoothing)."""
    h = _col(df, "high").values
    l = _col(df, "low").values
    c = _col(df, "close").values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = np.empty_like(tr)
    atr[:period] = np.nan
    atr[period - 1] = tr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Return the last ADX value."""
    h = _col(df, "high").values
    l = _col(df, "low").values
    c = _col(df, "close").values
    n = len(c)
    if n < period * 2:
        return 0.0

    prev_h = np.roll(h, 1); prev_h[0] = h[0]
    prev_l = np.roll(l, 1); prev_l[0] = l[0]
    prev_c = np.roll(c, 1); prev_c[0] = c[0]

    up_move   = h - prev_h
    down_move = prev_l - l
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))

    def wilder_smooth(arr, p):
        out = np.empty(len(arr))
        out[:p] = np.nan
        out[p - 1] = arr[:p].mean()
        a = 1.0 / p
        for i in range(p, len(arr)):
            out[i] = out[i - 1] * (1 - a) + arr[i] * a
        return out

    atr_s    = wilder_smooth(tr, period)
    plus_di  = 100 * wilder_smooth(plus_dm,  period) / np.where(atr_s == 0, np.nan, atr_s)
    minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr_s == 0, np.nan, atr_s)
    dx_denom = plus_di + minus_di
    dx       = np.where(dx_denom == 0, 0.0, 100 * np.abs(plus_di - minus_di) / dx_denom)
    adx_arr  = wilder_smooth(dx, period)
    valid    = adx_arr[~np.isnan(adx_arr)]
    return float(valid[-1]) if len(valid) else 0.0


def _linreg_slope(values: np.ndarray) -> float:
    """Ordinary-least-squares slope over the given array."""
    if len(values) < 2:
        return 0.0
    x    = np.arange(len(values), dtype=float)
    xm   = x.mean()
    ym   = values.mean()
    denom = ((x - xm) ** 2).sum()
    return float(((x - xm) * (values - ym)).sum() / denom) if denom != 0 else 0.0


def _swing_highs(high: np.ndarray, order: int = 3) -> np.ndarray:
    """
    Return indices of swing highs.
    A swing high at index i requires high[i] to be the maximum
    in the window [i-order, i+order].
    """
    indices = []
    for i in range(order, len(high) - order):
        window = high[i - order: i + order + 1]
        if high[i] == window.max():
            indices.append(i)
    return np.array(indices, dtype=int)


# ---------------------------------------------------------------------------
# Resistance cluster detection (DBSCAN with single-linkage fallback)
# ---------------------------------------------------------------------------

def _cluster_levels(
    prices: np.ndarray,
    tolerance_pct: float,
) -> List[Tuple[float, int]]:
    """
    Cluster price levels. Returns list of (centroid, count) sorted descending.
    Uses DBSCAN when sklearn is available; falls back to single-linkage.
    """
    if len(prices) == 0:
        return []

    try:
        from sklearn.cluster import DBSCAN
        eps    = prices.mean() * tolerance_pct
        db     = DBSCAN(eps=eps, min_samples=1).fit(prices.reshape(-1, 1))
        labels = db.labels_
        clusters = []
        for lbl in set(labels):
            if lbl == -1:
                continue
            members = prices[labels == lbl]
            clusters.append((float(members.mean()), int(len(members))))
        clusters.sort(key=lambda x: x[0], reverse=True)
        return clusters
    except ImportError:
        pass

    # Single-linkage fallback
    sorted_p = np.sort(prices)[::-1]
    clusters: List[Tuple[float, int]] = []
    for p in sorted_p:
        merged = False
        for i, (centroid, count) in enumerate(clusters):
            if abs(p - centroid) / centroid <= tolerance_pct:
                new_centroid = (centroid * count + p) / (count + 1)
                clusters[i] = (new_centroid, count + 1)
                merged = True
                break
        if not merged:
            clusters.append((float(p), 1))
    clusters.sort(key=lambda x: x[0], reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def _detect_structure(
    df: pd.DataFrame,
    cfg: dict,
    symbol: str,
) -> dict:
    """
    Detect valid breakout structure on a single timeframe.

    Returns dict with keys:
        valid_structure    : bool
        primary_resistance : float | None
        touches            : int
        compression        : bool
        vol_contraction    : bool
        atr_value          : float
        adx_value          : float
        rejection_reason   : str | None
    """
    high   = _col(df, "high").values
    low    = _col(df, "low").values
    close  = _col(df, "close").values
    volume = _col(df, "volume").values
    n      = len(close)

    result = dict(
        valid_structure=False,
        primary_resistance=None,
        touches=0,
        compression=False,
        vol_contraction=False,
        atr_value=0.0,
        adx_value=0.0,
        rejection_reason=None,
    )

    # ── ADX filter ────────────────────────────────────────────────
    adx_val = _adx(df)
    result["adx_value"] = adx_val
    if adx_val < cfg.get("adx_threshold", 20):
        result["rejection_reason"] = f"adx_too_low({adx_val:.1f})"
        return result

    # ── Average volume filter ─────────────────────────────────────
    avg_vol = float(np.median(volume[-20:])) if n >= 20 else float(volume.mean())
    if avg_vol < cfg.get("min_volume_24h", 50_000):
        result["rejection_reason"] = f"avg_volume_too_low({avg_vol:.0f})"
        return result

    # ── ATR ───────────────────────────────────────────────────────
    atr_arr = _atr(df)
    valid_atr = atr_arr[~np.isnan(atr_arr)]
    result["atr_value"] = float(valid_atr[-1]) if len(valid_atr) else 0.0

    # ── Swing highs → resistance clusters ────────────────────────
    sh_idx = _swing_highs(high, order=3)
    if len(sh_idx) < cfg.get("min_touches", 2):
        result["rejection_reason"] = "insufficient_swing_highs"
        return result

    tol      = cfg.get("resistance_tolerance_pct", 0.005)
    clusters = _cluster_levels(high[sh_idx], tolerance_pct=tol)

    resistance  = None
    touch_count = 0
    for centroid, count in clusters:
        if centroid > close[-1] * (1 - tol) and count >= cfg.get("min_touches", 2):
            resistance  = centroid
            touch_count = count
            break

    if resistance is None:
        result["rejection_reason"] = "no_valid_resistance_cluster"
        return result

    result["primary_resistance"] = resistance
    result["touches"]            = touch_count

    # ── Fake breakout rejection ───────────────────────────────────
    max_fakes  = cfg.get("max_fake_breakouts", 2)
    fake_count = 0
    M = 3
    for i in range(max(0, n - 50), n - M - 1):
        if close[i] > resistance:
            if any(close[i + 1: i + 1 + M] < resistance):
                fake_count += 1
    if fake_count > max_fakes:
        result["rejection_reason"] = f"too_many_fake_breakouts({fake_count})"
        return result

    # ── Compression ───────────────────────────────────────────────
    lookback    = min(20, n)
    ranges      = high[-lookback:] - low[-lookback:]
    range_slope = _linreg_slope(ranges)
    low_slope   = _linreg_slope(low[-lookback:])
    range_ratio = (ranges[-1] / ranges[0]) if ranges[0] > 0 else 1.0
    result["compression"] = (
        range_slope < 0
        and low_slope > 0
        and range_ratio < cfg.get("compression_range_ratio", 0.7)
    )

    # ── Volatility contraction ────────────────────────────────────
    atr_window = atr_arr[-10:]
    atr_window = atr_window[~np.isnan(atr_window)]
    result["vol_contraction"] = (
        len(atr_window) >= 5
        and _linreg_slope(atr_window) < cfg.get("atr_contraction_slope", -0.1)
    )

    # ── Volume behaviour ──────────────────────────────────────────
    vol_window = volume[-20:] if n >= 20 else volume
    vol_mean   = vol_window.mean()
    vol_cv     = float(vol_window.std() / vol_mean) if vol_mean > 0 else 1.0
    vol_slope  = _linreg_slope(vol_window)
    if not (vol_cv < cfg.get("volume_cv_threshold", 0.3) or vol_slope < 0):
        result["rejection_reason"] = f"volume_erratic(cv={vol_cv:.2f})"
        return result

    result["valid_structure"] = True
    return result


# ---------------------------------------------------------------------------
# Entry detection helpers
# ---------------------------------------------------------------------------

def _detect_breakout_entry(
    df: pd.DataFrame,
    resistance: float,
    cfg: dict,
) -> Optional[Tuple[str, float, dict]]:
    """
    Direct breakout entry: close above resistance with size and volume constraints.
    Returns (entry_type, confidence, extra_meta) or None.
    """
    close  = _col(df, "close").values
    volume = _col(df, "volume").values

    size_pct = (close[-1] - resistance) / resistance
    min_pct  = cfg.get("breakout_min_pct", 0.005)
    max_pct  = cfg.get("breakout_max_pct", 0.05)

    if not (min_pct <= size_pct <= max_pct):
        return None

    median_vol   = float(np.median(volume[-21:-1])) if len(volume) > 20 else float(volume[:-1].mean())
    vol_ratio    = volume[-1] / median_vol if median_vol > 0 else 0.0
    spike_factor = cfg.get("volume_spike_factor", 1.5)

    if vol_ratio < spike_factor:
        return None

    size_score = min(size_pct / max_pct, 1.0)
    vol_score  = min((vol_ratio - spike_factor) / spike_factor, 1.0)
    confidence = float(np.clip(0.5 * size_score + 0.5 * vol_score, 0.0, 1.0))

    return ("breakout", confidence, {
        "breakout_size_pct": round(size_pct, 6),
        "volume_ratio":      round(vol_ratio, 3),
    })


def _detect_retest_entry(
    df: pd.DataFrame,
    resistance: float,
    cfg: dict,
) -> Optional[Tuple[str, float, dict]]:
    """
    Retest entry: price returns to resistance after a confirmed breakout.
    Confirmation: long lower wick OR bullish engulfing.
    Returns (entry_type, confidence, extra_meta) or None.
    """
    close  = _col(df, "close").values
    high   = _col(df, "high").values
    low    = _col(df, "low").values
    open_  = _col(df, "open").values

    lookback  = cfg.get("retest_lookback_bars", 5)
    tolerance = cfg.get("retest_tolerance_pct", 0.005)
    n         = len(close)

    if n < lookback + 2:
        return None

    if not any(close[-(lookback + 1):-1] > resistance):
        return None

    dist_pct = (close[-1] - resistance) / resistance
    if not (-tolerance <= dist_pct <= tolerance):
        return None

    conf_score    = 0.0
    confirmations = []

    bar_range = high[-1] - low[-1]
    if bar_range > 0:
        lower_wick = (min(open_[-1], close[-1]) - low[-1]) / bar_range
        if lower_wick > cfg.get("retest_wick_ratio", 0.6):
            conf_score += 0.5
            confirmations.append(f"long_lower_wick({lower_wick:.2f})")

    if n >= 2 and close[-1] > high[-2] and open_[-1] < close[-2]:
        conf_score += 0.5
        confirmations.append("bullish_engulfing")

    if conf_score == 0.0:
        return None

    return ("retest", float(np.clip(conf_score, 0.0, 1.0)), {
        "retest_dist_pct": round(dist_pct, 6),
        "confirmations":   confirmations,
    })


def _detect_continuation_entry(
    df: pd.DataFrame,
    resistance: float,
    cfg: dict,
) -> Optional[Tuple[str, float, dict]]:
    """
    Continuation entry: strong breakout + shallow pullback + new local high break.
    Returns (entry_type, confidence, extra_meta) or None.
    """
    close    = _col(df, "close").values
    high     = _col(df, "high").values
    low      = _col(df, "low").values

    strong_pct = cfg.get("strong_breakout_pct", 0.02)
    max_depth  = cfg.get("max_pullback_depth", 0.38)
    lookback   = cfg.get("retest_lookback_bars", 5)
    n          = len(close)

    if n < lookback + 3:
        return None

    breakout_bar_idx = None
    breakout_high    = None
    for i in range(n - lookback - 1, n - 1):
        if (close[i] - resistance) / resistance >= strong_pct:
            breakout_bar_idx = i
            breakout_high    = high[i]
            break

    if breakout_bar_idx is None or breakout_high is None:
        return None

    pullback_low   = float(low[breakout_bar_idx:].min())
    breakout_range = breakout_high - resistance

    if breakout_range <= 0:
        return None

    depth = (breakout_high - pullback_low) / breakout_range
    if depth >= max_depth:
        return None

    if close[-1] <= float(high[breakout_bar_idx:].max()):
        return None

    confidence = float(np.clip(1.0 - (depth / max_depth), 0.0, 1.0))
    return ("continuation", confidence, {
        "breakout_high":  round(breakout_high, 8),
        "pullback_low":   round(pullback_low, 8),
        "pullback_depth": round(depth, 4),
    })


# ---------------------------------------------------------------------------
# Multi-timeframe confirmation
# ---------------------------------------------------------------------------

def _mtf_confirm(
    htf_df: pd.DataFrame,
    current_price: float,
    cfg: dict,
    symbol: str,
) -> Tuple[bool, dict]:
    """
    Validate HTF structure and check for clear overhead resistance.
    Returns (is_confirmed, htf_meta).
    """
    offset_pct = cfg.get("htf_resistance_offset_pct", 0.02)
    htf_struct = _detect_structure(htf_df, cfg, symbol=f"{symbol}[HTF]")

    htf_meta = {
        "htf_valid_structure":  htf_struct["valid_structure"],
        "htf_resistance":       htf_struct["primary_resistance"],
        "htf_adx":              htf_struct["adx_value"],
        "htf_rejection_reason": htf_struct.get("rejection_reason"),
    }

    if not htf_struct["valid_structure"]:
        return False, htf_meta

    htf_resistance = htf_struct["primary_resistance"]
    if htf_resistance is None:
        return False, htf_meta

    clearance_required = current_price * (1 + offset_pct)
    if htf_resistance <= clearance_required:
        htf_meta["htf_block_reason"] = (
            f"htf_resistance_too_close({htf_resistance:.6f} <= {clearance_required:.6f})"
        )
        return False, htf_meta

    return True, htf_meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_breakout(
    ltf_df,
    htf_df,
    symbol: str,
    config: dict,
) -> BreakoutSignal:
    """
    Pure function — no side effects.

    Args:
        ltf_df  : Lower-timeframe OHLCV (polars or pandas, ≥ 200 bars recommended)
        htf_df  : Higher-timeframe OHLCV (polars or pandas, ≥ 200 bars recommended)
        symbol  : Trading pair string, used only for metadata/logging
        config  : Strategy parameter dict — accepts flat dict or {"breakout": {...}}

    Returns:
        BreakoutSignal with triggered=True and entry_type set,
        or triggered=False with rejection_reason in metadata.

    Raises:
        ValueError: on invalid or insufficient input data.
    """
    cfg = config.get("breakout", config)

    ltf = _to_pandas(ltf_df)
    htf = _to_pandas(htf_df)

    for label, frame in [("ltf_df", ltf), ("htf_df", htf)]:
        for col in ("open", "high", "low", "close", "volume"):
            if col not in frame.columns:
                raise ValueError(f"Column '{col}' missing from {label}.")
        if len(frame) < 30:
            raise ValueError(f"{label} must contain at least 30 bars (got {len(frame)}).")

    current_price = float(_col(ltf, "close").iloc[-1])

    # Step 1 — LTF structure
    structure = _detect_structure(ltf, cfg, symbol)

    if not structure["valid_structure"]:
        return BreakoutSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={
                "symbol":           symbol,
                "rejection_reason": structure["rejection_reason"],
                "atr_value":        structure["atr_value"],
                "adx_value":        structure["adx_value"],
            },
        )

    resistance  = structure["primary_resistance"]
    touch_count = structure["touches"]

    # Step 2 — MTF confirmation
    mtf_ok, htf_meta = _mtf_confirm(htf, current_price, cfg, symbol)

    if not mtf_ok:
        return BreakoutSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={
                "symbol":                symbol,
                "rejection_reason":      "mtf_confirmation_failed",
                "resistance":            resistance,
                "touch_count":           touch_count,
                "compression_detected":  structure["compression"],
                "volatility_contraction":structure["vol_contraction"],
                "atr_value":             structure["atr_value"],
                "adx_value":             structure["adx_value"],
                **htf_meta,
            },
        )

    # Step 3 — Entry timing (retest preferred > continuation > direct breakout)
    entry_result = (
        _detect_retest_entry(ltf, resistance, cfg)
        or _detect_continuation_entry(ltf, resistance, cfg)
        or _detect_breakout_entry(ltf, resistance, cfg)
    )

    base_meta = {
        "symbol":                symbol,
        "resistance":            round(resistance, 8),
        "touch_count":           touch_count,
        "compression_detected":  structure["compression"],
        "volatility_contraction":structure["vol_contraction"],
        "atr_value":             round(structure["atr_value"], 8),
        "adx_value":             round(structure["adx_value"], 2),
        **htf_meta,
    }

    if entry_result is None:
        return BreakoutSignal(
            triggered=False,
            entry_type=None,
            confidence=0.0,
            metadata={**base_meta, "rejection_reason": "no_entry_condition_met"},
        )

    entry_type, confidence, extra_meta = entry_result

    return BreakoutSignal(
        triggered=True,
        entry_type=entry_type,
        confidence=round(confidence, 4),
        metadata={**base_meta, **extra_meta},
    )
