"""
backtest/backtester.py
Regime-Aware Backtesting Module — v2

Walk-forward simulation: detect regime bar-by-bar, route to grid or strategy stack.
Outputs: Strategy PnL, Grid PnL, Win Rate, Max Drawdown, Regime breakdown.

Usage:
    python -m backtest.backtester --symbol DOGE/USDT --timeframe 15m --bars 1000
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.regime import Regime, detect_regime
from core.resolver import resolve
from core.risk import RiskManager
from strategies import breakout, trend, momentum, volume, mean_reversion
from strategies.grid import GridState, build_grid, detect_breakout_for_grid, get_triggered_levels, should_exit_grid

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    strategy: str
    signal: str
    regime: str
    entry_bar: int
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    exit_price: Optional[float] = None
    exit_bar:   Optional[int]   = None
    pnl:        float = 0.0
    outcome:    str   = "OPEN"


@dataclass
class GridTrade:
    symbol: str
    bar: int
    side: str
    fill_price: float
    quantity: float
    pnl: float = 0.0
    closed: bool = False


@dataclass
class BacktestResult:
    symbol: str
    total_trades: int   = 0
    wins:         int   = 0
    losses:       int   = 0
    total_pnl:    float = 0.0
    win_rate:     float = 0.0
    max_drawdown: float = 0.0
    grid_pnl:     float = 0.0
    strategy_pnl: float = 0.0
    regime_counts: Dict[str, int] = field(default_factory=lambda: {"TRENDING": 0, "RANGING": 0, "UNCLEAR": 0})
    trades:       List[Trade]     = field(default_factory=list)
    grid_trades:  List[GridTrade] = field(default_factory=list)

    def summarize(self):
        if self.total_trades > 0:
            self.win_rate = round(self.wins / self.total_trades * 100, 2)
        self.max_drawdown = round(self._compute_drawdown(), 4)

    def _compute_drawdown(self) -> float:
        all_pnl = [(t.entry_bar, t.pnl) for t in self.trades] + [(g.bar, g.pnl) for g in self.grid_trades if g.closed]
        all_pnl.sort(key=lambda x: x[0])
        peak = cumulative = max_dd = 0.0
        for _, pnl in all_pnl:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            max_dd = max(max_dd, peak - cumulative)
        return max_dd


class Backtester:
    def __init__(self, config: dict):
        self.config = config
        self.sp = config.get("strategy_params", {})
        self.rc = config.get("regime", {})
        self.gc = config.get("grid", {})
        self.risk_manager = RiskManager(
            risk_per_trade_pct=config.get("risk_per_trade_pct", 1.5),
            min_risk_reward=config.get("min_risk_reward", 2.0),
            max_drawdown_pct=config.get("max_drawdown_pct", 10.0),
            grid_max_loss_pct=config.get("grid", {}).get("grid_max_loss_pct", 5.0),
        )

    def run(self, symbol: str, df: pd.DataFrame, initial_capital: float = 10_000.0) -> BacktestResult:
        result   = BacktestResult(symbol=symbol)
        capital  = initial_capital
        self.risk_manager.set_start_capital(capital)

        open_trade: Optional[Trade]      = None
        active_grid: Optional[GridState] = None
        open_grid_fills: List[GridTrade] = []

        min_bars = max(self.rc.get("ma_slow_period", 50) + 10, 60)

        for i in range(min_bars, len(df)):
            slice_df      = df.iloc[:i]
            bar           = df.iloc[i]
            current_price = float(bar["close"])

            regime, _ = detect_regime(slice_df, **self.rc)
            result.regime_counts[regime.value] += 1

            # Close strategy trade on SL/TP hit
            if open_trade is not None:
                closed = self._check_exit(open_trade, bar, i)
                if closed:
                    capital += closed.pnl
                    self.risk_manager.record_pnl(closed.pnl)
                    result.trades.append(closed)
                    result.total_trades += 1
                    result.wins   += 1 if closed.outcome == "WIN"  else 0
                    result.losses += 1 if closed.outcome == "LOSS" else 0
                    result.total_pnl    += closed.pnl
                    result.strategy_pnl += closed.pnl
                    open_trade = None
                else:
                    continue

            # Check grid exit conditions
            if active_grid is not None:
                exit_now, reason = should_exit_grid(
                    active_grid, current_price, regime,
                    breakout_detected=detect_breakout_for_grid(slice_df),
                    global_loss_limit_hit=self.risk_manager.is_grid_loss_limit_hit(active_grid),
                )
                if exit_now:
                    for gfill in open_grid_fills:
                        if not gfill.closed:
                            gfill.pnl = (current_price - gfill.fill_price if gfill.side == "BUY"
                                         else gfill.fill_price - current_price) * gfill.quantity
                            gfill.closed = True
                            capital += gfill.pnl
                            result.grid_trades.append(gfill)
                            result.grid_pnl += gfill.pnl
                    active_grid, open_grid_fills = None, []
                    continue

            # RANGING path
            if regime == Regime.RANGING:
                if active_grid is None and open_trade is None and not detect_breakout_for_grid(slice_df):
                    active_grid     = build_grid(symbol, slice_df, capital, **{k: self.gc.get(k, v) for k, v in
                                                  [("num_levels", 10), ("risk_per_level_pct", 0.3), ("lookback_bars", 50)]})
                    open_grid_fills = []
                if active_grid is not None:
                    for level in get_triggered_levels(active_grid, current_price, self.gc.get("level_tolerance_pct", 0.002)):
                        gfill = GridTrade(symbol=symbol, bar=i, side=level.side, fill_price=level.price, quantity=level.quantity)
                        open_grid_fills.append(gfill)
                        level.filled = True
                        active_grid.total_invested += level.price * level.quantity
                continue

            if regime == Regime.UNCLEAR:
                continue

            # TRENDING — close any active grid
            if active_grid is not None:
                for gfill in open_grid_fills:
                    if not gfill.closed:
                        gfill.pnl = (current_price - gfill.fill_price if gfill.side == "BUY"
                                     else gfill.fill_price - current_price) * gfill.quantity
                        gfill.closed = True
                        capital += gfill.pnl
                        result.grid_trades.append(gfill)
                        result.grid_pnl += gfill.pnl
                active_grid, open_grid_fills = None, []

            if open_trade is not None:
                continue

            signals = self._collect_signals(symbol, slice_df)
            winner  = resolve(signals) if signals else None
            if winner is None:
                continue

            next_idx = i + 1
            if next_idx >= len(df):
                break

            entry_price = float(df.iloc[next_idx]["open"])
            params = self.risk_manager.calculate(
                capital=capital, entry_price=entry_price,
                side=winner["signal"], atr=RiskManager.compute_atr(slice_df),
            )
            if params is None:
                continue

            open_trade = Trade(
                symbol=symbol, strategy=winner["strategy"], signal=winner["signal"],
                regime=regime.value, entry_bar=next_idx, entry_price=entry_price,
                stop_loss=params.stop_loss, take_profit=params.take_profit, quantity=params.quantity,
            )

        # Force-close remaining open trade
        if open_trade is not None:
            lp = float(df["close"].iloc[-1])
            open_trade.exit_price = lp
            open_trade.exit_bar   = len(df) - 1
            open_trade.pnl = (lp - open_trade.entry_price if open_trade.signal == "BUY"
                              else open_trade.entry_price - lp) * open_trade.quantity
            open_trade.outcome = "WIN" if open_trade.pnl > 0 else "LOSS"
            result.trades.append(open_trade)
            result.total_trades += 1
            result.wins   += 1 if open_trade.outcome == "WIN"  else 0
            result.losses += 1 if open_trade.outcome == "LOSS" else 0
            result.total_pnl    += open_trade.pnl
            result.strategy_pnl += open_trade.pnl

        # Force-close remaining grid fills
        if open_grid_fills:
            lp = float(df["close"].iloc[-1])
            for gfill in open_grid_fills:
                if not gfill.closed:
                    gfill.pnl = (lp - gfill.fill_price if gfill.side == "BUY"
                                 else gfill.fill_price - lp) * gfill.quantity
                    gfill.closed = True
                    result.grid_trades.append(gfill)
                    result.grid_pnl += gfill.pnl

        result.total_pnl += result.grid_pnl
        result.summarize()
        return result

    def _collect_signals(self, symbol: str, df) -> list:
        signals = []
        sp = self.sp
        if sp.get("breakout", {}).get("enabled", True):
            p = sp.get("breakout", {})
            sig = breakout.generate_signal(symbol, df,
                lookback_bars=p.get("lookback_bars", 20),
                breakout_buffer_pct=p.get("breakout_buffer_pct", 0.002),
                volume_factor=p.get("volume_factor", 1.5),
                consolidation_bars=p.get("consolidation_bars", 3),
                min_rr=p.get("min_rr", 2.0))
            if sig: signals.append(sig)
        if sp.get("trend", {}).get("enabled", True):
            p = sp.get("trend", {})
            sig = trend.generate_signal(symbol, df, ma_fast=p.get("ma_fast", 10), ma_slow=p.get("ma_slow", 50))
            if sig: signals.append(sig)
        if sp.get("momentum", {}).get("enabled", True):
            p = sp.get("momentum", {})
            sig = momentum.generate_signal(symbol, df, rsi_period=p.get("rsi_period", 14),
                rsi_buy_threshold=p.get("rsi_buy_threshold", 52), rsi_sell_threshold=p.get("rsi_sell_threshold", 48))
            if sig: signals.append(sig)
        if sp.get("volume_spike", {}).get("enabled", True):
            p = sp.get("volume_spike", {})
            sig = volume.generate_signal(symbol, df, volume_ma_period=p.get("volume_ma_period", 20),
                volume_multiplier=p.get("volume_multiplier", 2.0))
            if sig: signals.append(sig)
        if sp.get("mean_reversion", {}).get("enabled", True):
            p = sp.get("mean_reversion", {})
            sig = mean_reversion.generate_signal(symbol, df, bb_period=p.get("bb_period", 20),
                bb_std=p.get("bb_std", 2.0), rsi_period=p.get("rsi_period", 14),
                rsi_oversold=p.get("rsi_oversold", 30), rsi_overbought=p.get("rsi_overbought", 70))
            if sig: signals.append(sig)
        return signals

    def _check_exit(self, trade: Trade, bar, bar_idx: int) -> Optional[Trade]:
        high, low = float(bar["high"]), float(bar["low"])
        if trade.signal == "BUY":
            if high >= trade.take_profit: return self._close(trade, trade.take_profit, bar_idx, "WIN")
            if low  <= trade.stop_loss:   return self._close(trade, trade.stop_loss,   bar_idx, "LOSS")
        else:
            if low  <= trade.take_profit: return self._close(trade, trade.take_profit, bar_idx, "WIN")
            if high >= trade.stop_loss:   return self._close(trade, trade.stop_loss,   bar_idx, "LOSS")
        return None

    @staticmethod
    def _close(trade: Trade, exit_price: float, bar_idx: int, outcome: str) -> Trade:
        trade.exit_price = exit_price
        trade.exit_bar   = bar_idx
        trade.outcome    = outcome
        trade.pnl = (exit_price - trade.entry_price if trade.signal == "BUY"
                     else trade.entry_price - exit_price) * trade.quantity
        return trade


def _print_result(r: BacktestResult):
    sep = "=" * 60
    print(f"\n{sep}\n  BACKTEST — {r.symbol}\n{sep}")
    print(f"  Strategy trades : {r.total_trades}  (W={r.wins} L={r.losses}  WR={r.win_rate}%)")
    print(f"  Strategy PnL    : {r.strategy_pnl:+.4f} USDT")
    print(f"  Grid fills      : {len(r.grid_trades)}")
    print(f"  Grid PnL        : {r.grid_pnl:+.4f} USDT")
    print(f"  TOTAL PnL       : {r.total_pnl:+.4f} USDT")
    print(f"  Max Drawdown    : {r.max_drawdown:.4f} USDT")
    print(f"  Regime counts   : {r.regime_counts}\n{sep}")
    if r.trades:
        print("\n  Last 8 strategy trades:")
        print(f"  {'#':<3} {'Strategy':<15} {'Rgm':<9} {'S':<5} {'Entry':>10} {'Exit':>10} {'PnL':>10} Result")
        for idx, t in enumerate(r.trades[-8:], 1):
            ep = f"{t.exit_price:.6f}" if t.exit_price else "OPEN"
            print(f"  {idx:<3} {t.strategy:<15} {t.regime:<9} {t.signal:<5} {t.entry_price:>10.6f} {ep:>10} {t.pnl:>+10.4f} {t.outcome}")
    if r.grid_trades:
        print(f"\n  Last 8 grid fills:")
        for idx, g in enumerate(r.grid_trades[-8:], 1):
            print(f"  {idx:<3} {g.side:<5} {g.fill_price:>10.6f} {g.pnl:>+10.4f}")
    print()


if __name__ == "__main__":
    import ccxt
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="DOGE/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--bars",      type=int,   default=500)
    parser.add_argument("--capital",   type=float, default=10_000.0)
    parser.add_argument("--config",    default="config.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_comment")}

    exchange = ccxt.bybit({"options": {"defaultType": "spot"}})
    raw = exchange.fetch_ohlcv(args.symbol, timeframe=args.timeframe, limit=args.bars)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df  = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    df.set_index("timestamp", inplace=True)

    _print_result(Backtester(cfg).run(args.symbol, df, initial_capital=args.capital))
