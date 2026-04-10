"""
core/risk.py
Risk Management Module.
- Dynamic position sizing (risk % of capital)
- SL/TP calculation (min R:R enforced)
- Global drawdown / emergency-stop tracking
- Per-grid emergency stop monitoring
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)
DEFAULT_ATR_MULTIPLIER = 1.5


@dataclass
class TradeParameters:
    quantity: float
    stop_loss: float
    take_profit: float
    risk_usd: float
    risk_pct: float


class RiskManager:
    def __init__(
        self,
        risk_per_trade_pct:    float = 1.5,
        min_risk_reward:       float = 2.0,
        max_concurrent_trades: int   = 5,
        max_drawdown_pct:      float = 10.0,
        grid_max_loss_pct:     float = 5.0,
    ):
        if not (0 < risk_per_trade_pct <= 5):
            raise ValueError("risk_per_trade_pct must be between 0 and 5.")
        if min_risk_reward < 1:
            raise ValueError("min_risk_reward must be >= 1.")

        self.risk_per_trade_pct    = risk_per_trade_pct
        self.min_risk_reward       = min_risk_reward
        self.max_concurrent_trades = max_concurrent_trades
        self.max_drawdown_pct      = max_drawdown_pct
        self.grid_max_loss_pct     = grid_max_loss_pct
        self._session_start_capital: Optional[float] = None
        self._session_realised_pnl: float = 0.0

    def set_start_capital(self, capital: float):
        if self._session_start_capital is None:
            self._session_start_capital = capital
            logger.info(f"[RiskManager] Session start capital: {capital:.2f} USDT")

    def record_pnl(self, pnl: float):
        self._session_realised_pnl += pnl

    def is_emergency_stop_triggered(self, current_capital: float) -> bool:
        if self._session_start_capital and self._session_start_capital > 0:
            drawdown_pct = (self._session_start_capital - current_capital) / self._session_start_capital * 100
            if drawdown_pct >= self.max_drawdown_pct:
                logger.critical(f"[RiskManager] ⛔ EMERGENCY STOP: drawdown={drawdown_pct:.2f}%")
                return True
        return False

    def is_grid_loss_limit_hit(self, grid_state) -> bool:
        if grid_state.total_invested <= 0:
            return False
        loss_pct = (-grid_state.realised_pnl / grid_state.total_invested) * 100
        if loss_pct >= self.grid_max_loss_pct:
            logger.warning(f"[RiskManager] Grid {grid_state.symbol} loss {loss_pct:.2f}% exceeds limit.")
            return True
        return False

    def can_open_trade(self, open_count: int) -> bool:
        if open_count >= self.max_concurrent_trades:
            logger.info(f"[RiskManager] Max trades reached ({open_count}/{self.max_concurrent_trades}).")
            return False
        return True

    def calculate(
        self, capital: float, entry_price: float, side: str,
        atr: Optional[float] = None, sl_price: Optional[float] = None,
    ) -> Optional[TradeParameters]:
        if capital <= 0 or entry_price <= 0:
            return None

        if sl_price is not None:
            sl_distance = abs(entry_price - sl_price)
        elif atr is not None and atr > 0:
            sl_distance = atr * DEFAULT_ATR_MULTIPLIER
            sl_price = entry_price - sl_distance if side == "BUY" else entry_price + sl_distance
        else:
            logger.error("[RiskManager] Must provide either sl_price or atr.")
            return None

        if sl_distance <= 0:
            return None

        risk_usd = capital * (self.risk_per_trade_pct / 100.0)
        quantity = risk_usd / sl_distance

        if quantity * entry_price > capital:
            quantity = capital / entry_price

        tp_distance = sl_distance * self.min_risk_reward
        if side == "BUY":
            take_profit, stop_loss = entry_price + tp_distance, sl_price
        else:
            take_profit, stop_loss = entry_price - tp_distance, sl_price

        return TradeParameters(
            quantity=round(quantity, 6), stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8), risk_usd=round(risk_usd, 4),
            risk_pct=self.risk_per_trade_pct,
        )

    @staticmethod
    def compute_atr(df, period: int = 14) -> float:
        high, low, close = df["high"], df["low"], df["close"]
        tr = (
            (high - low)
            .combine((high - close.shift(1)).abs(), max)
            .combine((low  - close.shift(1)).abs(), max)
        )
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
