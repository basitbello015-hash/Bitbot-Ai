"""
core/position_manager.py
Tracks open positions AND active grid states.
Enforces exactly one active trade (or grid) per coin.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)
STATE_FILE = "positions_state.json"


@dataclass
class Position:
    symbol: str
    side: str
    strategy: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    order_id: str
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PositionManager:
    def __init__(self, persist: bool = True):
        self._positions: Dict[str, Position] = {}
        self._grids: Dict[str, object] = {}
        self._persist = persist
        if persist:
            self._load_state()

    def has_open_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def has_active_grid(self, symbol: str) -> bool:
        return symbol in self._grids

    def is_occupied(self, symbol: str) -> bool:
        return self.has_open_position(symbol) or self.has_active_grid(symbol)

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def get_grid(self, symbol: str):
        return self._grids.get(symbol)

    def all_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def all_grids(self) -> dict:
        return dict(self._grids)

    def open_count(self) -> int:
        return len(self._positions) + len(self._grids)

    def open_position(self, position: Position) -> bool:
        if self.is_occupied(position.symbol):
            kind = "grid" if self.has_active_grid(position.symbol) else "position"
            logger.warning(f"[PositionManager] Rejected open_position for {position.symbol} — {kind} already active.")
            return False
        self._positions[position.symbol] = position
        logger.info(
            f"[PositionManager] Opened {position.side} on {position.symbol} via {position.strategy} | "
            f"entry={position.entry_price} qty={position.quantity} SL={position.stop_loss} TP={position.take_profit}"
        )
        if self._persist:
            self._save_state()
        return True

    def close_position(self, symbol: str) -> Optional[Position]:
        position = self._positions.pop(symbol, None)
        if position:
            logger.info(f"[PositionManager] Closed position on {symbol}.")
            if self._persist:
                self._save_state()
        return position

    def open_grid(self, grid_state) -> bool:
        symbol = grid_state.symbol
        if self.is_occupied(symbol):
            logger.warning(f"[PositionManager] Rejected open_grid for {symbol} — already occupied.")
            return False
        self._grids[symbol] = grid_state
        logger.info(
            f"[PositionManager] Grid activated for {symbol} | "
            f"range=[{grid_state.lower_bound:.6f}, {grid_state.upper_bound:.6f}] | levels={grid_state.num_levels}"
        )
        if self._persist:
            self._save_state()
        return True

    def close_grid(self, symbol: str, reason: str = "") -> Optional[object]:
        gs = self._grids.pop(symbol, None)
        if gs:
            gs.active = False
            logger.info(f"[PositionManager] Grid deactivated for {symbol}" + (f" — {reason}" if reason else ""))
            if self._persist:
                self._save_state()
        return gs

    def update_grid(self, symbol: str, grid_state) -> None:
        if symbol in self._grids:
            self._grids[symbol] = grid_state
            if self._persist:
                self._save_state()

    def _save_state(self):
        try:
            state = {
                "positions": {sym: asdict(pos) for sym, pos in self._positions.items()},
                "grids":     {sym: gs.to_dict() for sym, gs in self._grids.items()},
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"[PositionManager] Failed to save state: {e}")

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            from strategies.grid import GridState
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            if "positions" not in state and "grids" not in state:
                positions_data, grids_data = state, {}
            else:
                positions_data = state.get("positions", {})
                grids_data     = state.get("grids", {})
            for sym, d in positions_data.items():
                self._positions[sym] = Position(**d)
            for sym, d in grids_data.items():
                self._grids[sym] = GridState.from_dict(d)
            logger.info(
                f"[PositionManager] Restored {len(self._positions)} position(s) and {len(self._grids)} grid(s)."
            )
        except Exception as e:
            logger.error(f"[PositionManager] Failed to load state: {e}")
