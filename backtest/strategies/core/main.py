"""
main.py
Trading Bot Entry Point — v2 with Regime Detection + Grid Mode.

Decision flow per coin per cycle:
  1. Detect market regime (TRENDING / RANGING / UNCLEAR)
  2. UNCLEAR  → skip coin
  3. RANGING  → activate / continue Grid Mode (Plan B)
  4. TRENDING → run strategy stack in priority order
  5. One position OR one grid per coin at all times

Run:
    export BYBIT_API_KEY=your_key
    export BYBIT_API_SECRET=your_secret
    python main.py [--dry-run] [--config path/to/config.json]
"""

import argparse
import json
import logging
import sys
import time
from typing import List, Optional

from core.data import MarketData
from core.execution import ExecutionEngine, build_exchange
from core.position_manager import Position, PositionManager
from core.regime import Regime, detect_regime
from core.resolver import resolve
from core.risk import RiskManager
from strategies import breakout, trend, momentum, volume, mean_reversion
from strategies.grid import (
    GridState,
    build_grid,
    detect_breakout_for_grid,
    get_triggered_levels,
    should_exit_grid,
)

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log"),
    ],
)
logger = logging.getLogger("main")


# ════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_comment")}


# ════════════════════════════════════════════════════════════════════
# Coin validation
# ════════════════════════════════════════════════════════════════════

def validate_coins(
    coins: List[str],
    market_data: MarketData,
    min_price: float,
    max_price: float,
) -> List[str]:
    valid = []
    for symbol in coins:
        price = market_data.fetch_current_price(symbol)
        if price is None:
            logger.warning(f"[Validate] {symbol}: could not fetch price — skipping.")
            continue
        if min_price <= price <= max_price:
            logger.info(f"[Validate] ✓ {symbol}: ${price:.6f}")
            valid.append(symbol)
        else:
            logger.warning(f"[Validate] ✗ {symbol}: ${price:.6f} outside range — skipping.")
    return valid


# ════════════════════════════════════════════════════════════════════
# Strategy signal collection (TRENDING path only)
# ════════════════════════════════════════════════════════════════════

def collect_signals(symbol: str, df, sp: dict) -> list:
    """Evaluate all enabled strategies and return their signals."""
    signals = []

    if sp.get("breakout", {}).get("enabled", True):
        p = sp["breakout"]
        sig = breakout.generate_signal(
            symbol, df,
            lookback_bars=p.get("lookback_bars", 20),
            breakout_buffer_pct=p.get("breakout_buffer_pct", 0.002),
            volume_factor=p.get("volume_factor", 1.5),
            consolidation_bars=p.get("consolidation_bars", 3),
            min_rr=p.get("min_rr", 2.0),
        )
        if sig:
            signals.append(sig)

    if sp.get("trend", {}).get("enabled", True):
        p = sp["trend"]
        sig = trend.generate_signal(
            symbol, df,
            ma_fast=p.get("ma_fast", 10),
            ma_slow=p.get("ma_slow", 50),
        )
        if sig:
            signals.append(sig)

    if sp.get("momentum", {}).get("enabled", True):
        p = sp["momentum"]
        sig = momentum.generate_signal(
            symbol, df,
            rsi_period=p.get("rsi_period", 14),
            rsi_buy_threshold=p.get("rsi_buy_threshold", 52),
            rsi_sell_threshold=p.get("rsi_sell_threshold", 48),
        )
        if sig:
            signals.append(sig)

    if sp.get("volume_spike", {}).get("enabled", True):
        p = sp["volume_spike"]
        sig = volume.generate_signal(
            symbol, df,
            volume_ma_period=p.get("volume_ma_period", 20),
            volume_multiplier=p.get("volume_multiplier", 2.0),
        )
        if sig:
            signals.append(sig)

    if sp.get("mean_reversion", {}).get("enabled", True):
        p = sp["mean_reversion"]
        sig = mean_reversion.generate_signal(
            symbol, df,
            bb_period=p.get("bb_period", 20),
            bb_std=p.get("bb_std", 2.0),
            rsi_period=p.get("rsi_period", 14),
            rsi_oversold=p.get("rsi_oversold", 30),
            rsi_overbought=p.get("rsi_overbought", 70),
        )
        if sig:
            signals.append(sig)

    return signals


# ════════════════════════════════════════════════════════════════════
# Trade execution (strategy positions)
# ════════════════════════════════════════════════════════════════════

def execute_trade(
    signal: dict,
    entry_price: float,
    capital: float,
    risk_manager: RiskManager,
    position_manager: PositionManager,
    execution_engine: Optional[ExecutionEngine],
    dry_run: bool,
    market_data: MarketData,
    timeframe: str,
) -> bool:
    symbol = signal["coin"]
    side   = signal["signal"]

    # Use SL from signal if strategy provided it, else fall back to ATR
    sl_price = signal.get("stop_loss")
    atr = None
    if sl_price is None:
        df_atr = market_data.fetch_ohlcv(symbol, timeframe=timeframe, limit=50)
        atr = RiskManager.compute_atr(df_atr) if df_atr is not None else None

    params = risk_manager.calculate(
        capital=capital,
        entry_price=entry_price,
        side=side,
        atr=atr,
        sl_price=sl_price,
    )
    if params is None:
        logger.error(f"[Trade] Cannot compute params for {symbol}.")
        return False

    if dry_run:
        logger.info(
            f"[DRY RUN] {side} {params.quantity} {symbol} @ {entry_price:.6f} | "
            f"SL={params.stop_loss:.6f} TP={params.take_profit:.6f} | "
            f"risk={params.risk_usd:.2f} USDT | strategy={signal['strategy']}"
        )
        position = Position(
            symbol=symbol, side=side, strategy=signal["strategy"],
            entry_price=entry_price, quantity=params.quantity,
            stop_loss=params.stop_loss, take_profit=params.take_profit,
            order_id="DRY_RUN",
        )
        return position_manager.open_position(position)

    if side == "BUY":
        entry_id, sl_id, tp_id = execution_engine.execute_buy(
            symbol, params.quantity, params.stop_loss, params.take_profit
        )
    else:
        entry_id, sl_id, tp_id = execution_engine.execute_sell(
            symbol, params.quantity, params.stop_loss, params.take_profit
        )

    if entry_id is None:
        logger.error(f"[Trade] Entry order failed for {symbol}.")
        return False

    position = Position(
        symbol=symbol, side=side, strategy=signal["strategy"],
        entry_price=entry_price, quantity=params.quantity,
        stop_loss=params.stop_loss, take_profit=params.take_profit,
        order_id=entry_id,
    )
    return position_manager.open_position(position)


# ════════════════════════════════════════════════════════════════════
# Grid management
# ════════════════════════════════════════════════════════════════════

def handle_ranging_coin(
    symbol: str,
    df,
    current_price: float,
    current_regime: Regime,
    capital: float,
    grid_cfg: dict,
    risk_manager: RiskManager,
    position_manager: PositionManager,
    execution_engine: Optional[ExecutionEngine],
    dry_run: bool,
):
    # ── A. Existing grid ─────────────────────────────────────────
    if position_manager.has_active_grid(symbol):
        gs: GridState = position_manager.get_grid(symbol)

        breakout_now = detect_breakout_for_grid(df, lookback_bars=20, buffer_pct=0.002)
        global_risk_hit = risk_manager.is_grid_loss_limit_hit(gs)

        exit_now, reason = should_exit_grid(
            gs, current_price, regime=current_regime,
            breakout_detected=breakout_now,
            global_loss_limit_hit=global_risk_hit,
        )

        if exit_now:
            logger.info(f"[Grid] Closing grid on {symbol}: {reason}")
            position_manager.close_grid(symbol, reason=reason)
            return

        triggered = get_triggered_levels(
            gs, current_price,
            tolerance_pct=grid_cfg.get("level_tolerance_pct", 0.002),
        )
        for level in triggered:
            if dry_run:
                logger.info(
                    f"[Grid DRY RUN] {level.side} {level.quantity} {symbol} "
                    f"@ level {level.price:.6f} (current={current_price:.6f})"
                )
                level.filled = True
                cost = level.price * level.quantity
                gs.total_invested += cost if level.side == "BUY" else 0
            else:
                if level.side == "BUY":
                    eid, _, _ = execution_engine.execute_buy(
                        symbol, level.quantity,
                        stop_loss=gs.lower_bound * 0.99,
                        take_profit=gs.upper_bound,
                    )
                else:
                    eid, _, _ = execution_engine.execute_sell(
                        symbol, level.quantity,
                        stop_loss=gs.upper_bound * 1.01,
                        take_profit=gs.lower_bound,
                    )
                if eid:
                    level.order_id = eid
                    level.filled   = True
                    cost = level.price * level.quantity
                    gs.total_invested += cost if level.side == "BUY" else 0

        position_manager.update_grid(symbol, gs)
        return

    # ── B. Activate new grid ─────────────────────────────────────
    if not grid_cfg.get("enabled", True):
        return

    if position_manager.is_occupied(symbol):
        return

    breakout_present = detect_breakout_for_grid(df, lookback_bars=20, buffer_pct=0.002)
    if breakout_present:
        logger.info(f"[Grid] {symbol}: breakout detected — not activating grid.")
        return

    if not risk_manager.can_open_trade(position_manager.open_count()):
        return

    gs = build_grid(
        symbol=symbol, df=df, capital=capital,
        num_levels=grid_cfg.get("num_levels", 10),
        risk_per_level_pct=grid_cfg.get("risk_per_level_pct", 0.3),
        lookback_bars=grid_cfg.get("lookback_bars", 50),
    )
    if gs is None:
        return

    position_manager.open_grid(gs)


# ════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════

def run(config_path: str = "config.json", dry_run: bool = False):
    logger.info("=" * 65)
    logger.info("  Bybit Multi-Strategy + Grid Trading Bot  v2")
    logger.info(f"  Mode: {'DRY RUN (no orders placed)' if dry_run else '⚠️  LIVE TRADING'}")
    logger.info("=" * 65)

    cfg = load_config(config_path)

    exchange_cfg = cfg.get("exchange", {})
    testnet  = exchange_cfg.get("testnet", True)
    exchange = build_exchange(testnet=testnet)

    market_data       = MarketData(exchange)
    position_manager  = PositionManager(persist=True)
    risk_manager      = RiskManager(
        risk_per_trade_pct=cfg.get("risk_per_trade_pct", 1.5),
        min_risk_reward=cfg.get("min_risk_reward", 2.0),
        max_concurrent_trades=cfg.get("max_concurrent_trades", 5),
        max_drawdown_pct=cfg.get("max_drawdown_pct", 10.0),
        grid_max_loss_pct=cfg.get("grid", {}).get("grid_max_loss_pct", 5.0),
    )
    execution_engine  = ExecutionEngine(exchange) if not dry_run else None

    timeframe      = cfg.get("timeframe", "15m")
    loop_interval  = cfg.get("loop_interval_seconds", 60)
    sp             = cfg.get("strategy_params", {})
    price_filter   = cfg.get("price_filter", {"min_price_usd": 0.001, "max_price_usd": 2.0})
    regime_cfg     = cfg.get("regime", {})
    grid_cfg       = cfg.get("grid", {})

    raw_coins: List[str] = cfg.get("coins", [])
    logger.info(f"Configured coins ({len(raw_coins)}): {raw_coins}")

    active_coins = validate_coins(
        raw_coins, market_data,
        min_price=price_filter["min_price_usd"],
        max_price=price_filter["max_price_usd"],
    )
    if not active_coins:
        logger.error("No valid coins after price filter. Exiting.")
        sys.exit(1)

    logger.info(f"Active coins ({len(active_coins)}): {active_coins}")

    opening_capital = market_data.get_usdt_balance() if not dry_run else 10_000.0
    risk_manager.set_start_capital(opening_capital)

    cycle = 0
    while True:
        cycle += 1
        capital = market_data.get_usdt_balance() if not dry_run else 10_000.0

        logger.info(
            f"── Cycle {cycle} | Capital: {capital:.2f} USDT | "
            f"Open: {position_manager.open_count()} "
            f"(positions={len(position_manager.all_positions())} "
            f"grids={len(position_manager.all_grids())}) ──"
        )

        if risk_manager.is_emergency_stop_triggered(capital):
            logger.critical("⛔ Emergency stop triggered. Bot halted.")
            sys.exit(1)

        for symbol in active_coins:
            try:
                df = market_data.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
                if df is None:
                    continue

                current_price = float(df["close"].iloc[-1])

                if not (price_filter["min_price_usd"] <= current_price <= price_filter["max_price_usd"]):
                    logger.debug(f"[{symbol}] ${current_price:.6f} outside price filter.")
                    continue

                regime, metrics = detect_regime(df, **regime_cfg)

                logger.info(
                    f"[{symbol}] Regime={regime.value} | "
                    f"ADX={metrics.get('adx', '?')} | "
                    f"MA_sep={metrics.get('ma_sep_pct', '?')}% | "
                    f"Price=${current_price:.6f}"
                )

                if regime == Regime.UNCLEAR:
                    logger.debug(f"[{symbol}] Regime UNCLEAR — skipping.")
                    continue

                if regime == Regime.RANGING:
                    handle_ranging_coin(
                        symbol=symbol, df=df, current_price=current_price,
                        current_regime=regime, capital=capital,
                        grid_cfg=grid_cfg, risk_manager=risk_manager,
                        position_manager=position_manager,
                        execution_engine=execution_engine, dry_run=dry_run,
                    )
                    continue

                if position_manager.has_active_grid(symbol):
                    position_manager.close_grid(symbol, reason="regime_shift_to_trending")

                if position_manager.has_open_position(symbol):
                    logger.debug(f"[{symbol}] Strategy position open — skipping.")
                    continue

                if not risk_manager.can_open_trade(position_manager.open_count()):
                    logger.debug("Max concurrent trades reached — stopping scan.")
                    break

                signals = collect_signals(symbol, df, sp)
                if not signals:
                    logger.debug(f"[{symbol}] No signals this cycle.")
                    continue

                winner = resolve(signals)
                if winner is None:
                    continue

                logger.info(
                    f"[{symbol}] ✅ Signal: {winner['signal']} via "
                    f"{winner['strategy']} (strength={winner.get('strength', 'N/A')})"
                )

                execute_trade(
                    signal=winner, entry_price=current_price, capital=capital,
                    risk_manager=risk_manager, position_manager=position_manager,
                    execution_engine=execution_engine, dry_run=dry_run,
                    market_data=market_data, timeframe=timeframe,
                )

            except Exception as e:
                logger.exception(f"[{symbol}] Unhandled error: {e}")

        logger.info(f"── Cycle {cycle} done. Sleeping {loop_interval}s ──\n")
        time.sleep(loop_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bybit Multi-Strategy + Grid Trading Bot")
    parser.add_argument("--config",  default="config.json", help="Path to config.json")
    parser.add_argument("--dry-run", action="store_true", help="Paper mode: no real orders.")
    args = parser.parse_args()
    run(config_path=args.config, dry_run=args.dry_run)
