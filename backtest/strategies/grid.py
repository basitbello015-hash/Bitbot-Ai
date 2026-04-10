"""
strategies/grid.py
Grid Strategy — Special Mode / Plan B (regime-gated, not priority-ranked)

Activates only when: regime == RANGING AND no active position AND no breakout.
Divides recent high/low range into equally-spaced buy/sell levels.
Exits on: regime shift to TRENDING, price break outside range, or loss limit.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
STRATEGY_NAME = "Grid"


@dataclass
class GridLevel:
    price: float
    side: str
    quantity: float
    order_id: str = ""
    filled: bool = False


@dataclass
class GridState:
    symbol: str
    upper_bound: float
    lower_bound: float
    num_levels: int
    levels: List[GridLevel] = field(default_factory=list)
    active: bool = True
    total_invested: float = 0.0
    realised_pnl: float = 0.0

    def level_prices(self) -> List[float]:
        return [lv.price for lv in self.levels]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound, "num_levels": self.num_levels,
            "active": self.active, "total_invested": self.total_invested,
            "realised_pnl": self.realised_pnl,
            "levels": [
                {"price": lv.price, "side": lv.side, "quantity": lv.quantity,
                 "order_id": lv.order_id, "filled": lv.filled}
                for lv in self.levels
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GridState":
        levels = [GridLevel(**lv) for lv in d.pop("levels", [])]
        gs = cls(**d)
        gs.levels = levels
        return gs


def build_grid(
    symbol: str, df: pd.DataFrame, capital: float,
    num_levels: int = 10, risk_per_level_pct: float = 0.3, lookback_bars: int = 50,
) -> Optional[GridState]:
    if df is None or len(df) < lookback_bars:
        return None

    window        = df.iloc[-lookback_bars:]
    upper_bound   = float(window["high"].max())
    lower_bound   = float(window["low"].min())
    current_price = float(df["close"].iloc[-1])

    if upper_bound <= lower_bound:
        return None

    step        = (upper_bound - lower_bound) / (num_levels + 1)
    grid_prices = [round(lower_bound + step * i, 8) for i in range(1, num_levels + 1)]
    capital_per_level = capital * (risk_per_level_pct / 100.0)

    levels = []
    for price in grid_prices:
        quantity = round(capital_per_level / price, 6) if price > 0 else 0.0
        if quantity <= 0:
            continue
        side = "BUY" if price < current_price else "SELL"
        levels.append(GridLevel(price=price, side=side, quantity=quantity))

    if not levels:
        return None

    gs = GridState(symbol=symbol, upper_bound=upper_bound, lower_bound=lower_bound, num_levels=num_levels, levels=levels)
    logger.info(f"[Grid] Built grid for {symbol} | range=[{lower_bound:.6f}, {upper_bound:.6f}] | levels={len(levels)}")
    return gs


def should_exit_grid(
    gs: GridState, current_price: float, regime,
    breakout_detected: bool = False, global_loss_limit_hit: bool = False,
) -> Tuple[bool, str]:
    from core.regime import Regime
    if regime == Regime.TRENDING:
        return True, "regime_shift_to_trending"
    if current_price > gs.upper_bound:
        return True, f"price_broke_above_upper_bound({gs.upper_bound:.6f})"
    if current_price < gs.lower_bound:
        return True, f"price_broke_below_lower_bound({gs.lower_bound:.6f})"
    if breakout_detected:
        return True, "breakout_detected"
    if global_loss_limit_hit:
        return True, "global_risk_limit_reached"
    return False, ""


def get_triggered_levels(
    gs: GridState, current_price: float, tolerance_pct: float = 0.002,
) -> List[GridLevel]:
    if not gs.active:
        return []
    return [
        lv for lv in gs.levels
        if not lv.filled and abs(current_price - lv.price) / lv.price <= tolerance_pct
    ]


def detect_breakout_for_grid(
    df: pd.DataFrame, lookback_bars: int = 20, buffer_pct: float = 0.002,
) -> bool:
    if df is None or len(df) < lookback_bars + 1:
        return False
    window        = df.iloc[-(lookback_bars + 1):-1]
    current_close = float(df["close"].iloc[-1])
    resistance    = float(window["high"].max())
    support       = float(window["low"].min())
    return (
        current_close > resistance * (1 + buffer_pct)
        or current_close < support * (1 - buffer_pct)
    )
