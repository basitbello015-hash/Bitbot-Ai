"""
core/resolver.py
Priority & Conflict Resolver — TRENDING strategy stack only.
Grid is regime-gated in main.py, not priority-ranked here.

Priority (highest → lowest):
  1. Breakout
  2. TrendFollowing
  3. Momentum
  4. VolumeSpike
  5. MeanReversion
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STRATEGY_PRIORITY: List[str] = [
    "Breakout", "TrendFollowing", "Momentum", "VolumeSpike", "MeanReversion",
]
_PRIORITY_RANK: Dict[str, int] = {name: i for i, name in enumerate(STRATEGY_PRIORITY)}


def resolve(signals: List[Dict]) -> Optional[Dict]:
    if not signals:
        return None
    for s in signals:
        if s.get("strategy", "") not in _PRIORITY_RANK:
            raise ValueError(f"[Resolver] Unknown strategy '{s.get('strategy')}'. Valid: {STRATEGY_PRIORITY}")
    if len(signals) == 1:
        return signals[0]

    def sort_key(s):
        return (_PRIORITY_RANK[s["strategy"]], -(s.get("strength") or 0.0))

    signals_sorted = sorted(signals, key=sort_key)
    winner = signals_sorted[0]
    losers = [s["strategy"] for s in signals_sorted[1:]]
    logger.info(f"[Resolver] {winner['coin']} — winner='{winner['strategy']}' ({winner['signal']}) | suppressed={losers}")
    return winner
