"""
core/execution.py
Execution Engine — places spot market orders with SL and TP on Bybit.
API keys read from environment variables: BYBIT_API_KEY, BYBIT_API_SECRET
"""

import logging
import os
from typing import Optional, Tuple
import ccxt

logger = logging.getLogger(__name__)


def build_exchange(testnet: bool = True) -> ccxt.bybit:
    api_key    = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        raise EnvironmentError(
            "Missing Bybit credentials. Set BYBIT_API_KEY and BYBIT_API_SECRET."
        )

    exchange = ccxt.bybit({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "spot"},
    })

    if testnet:
        exchange.set_sandbox_mode(True)
        logger.info("[Execution] Running in TESTNET mode.")
    else:
        logger.warning("[Execution] Running in LIVE mode. Real funds at risk.")

    exchange.load_markets()
    return exchange


class ExecutionEngine:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def execute_buy(
        self, symbol: str, quantity: float, stop_loss: float, take_profit: float
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        entry_id = self._place_market_order(symbol, "buy", quantity)
        if entry_id is None:
            return None, None, None
        sl_id = self._place_stop_loss(symbol, "sell", quantity, stop_loss)
        tp_id = self._place_take_profit(symbol, "sell", quantity, take_profit)
        return entry_id, sl_id, tp_id

    def execute_sell(
        self, symbol: str, quantity: float, stop_loss: float, take_profit: float
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        entry_id = self._place_market_order(symbol, "sell", quantity)
        if entry_id is None:
            return None, None, None
        sl_id = self._place_stop_loss(symbol, "buy", quantity, stop_loss)
        tp_id = self._place_take_profit(symbol, "buy", quantity, take_profit)
        return entry_id, sl_id, tp_id

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"[Execution] cancel_order failed for {order_id}: {e}")
            return False

    def fetch_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        try:
            return self.exchange.fetch_order(order_id, symbol).get("status")
        except Exception as e:
            logger.error(f"[Execution] fetch_order_status error: {e}")
            return None

    def _place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[str]:
        try:
            order = self.exchange.create_order(symbol=symbol, type="market", side=side, amount=quantity)
            logger.info(f"[Execution] Market {side.upper()} {quantity} {symbol} → {order['id']}")
            return order["id"]
        except ccxt.InsufficientFunds as e:
            logger.error(f"[Execution] Insufficient funds for {symbol}: {e}")
        except ccxt.InvalidOrder as e:
            logger.error(f"[Execution] Invalid order for {symbol}: {e}")
        except Exception as e:
            logger.error(f"[Execution] Unexpected error: {e}")
        return None

    def _place_stop_loss(self, symbol: str, side: str, quantity: float, sl_price: float) -> Optional[str]:
        try:
            order = self.exchange.create_order(
                symbol=symbol, type="stop", side=side, amount=quantity, price=sl_price,
                params={"stopPrice": sl_price, "triggerBy": "LastPrice"},
            )
            logger.info(f"[Execution] SL {side.upper()} {quantity} {symbol} @ {sl_price} → {order['id']}")
            return order["id"]
        except Exception as e:
            logger.error(f"[Execution] Failed to place SL for {symbol}: {e}")
        return None

    def _place_take_profit(self, symbol: str, side: str, quantity: float, tp_price: float) -> Optional[str]:
        try:
            order = self.exchange.create_order(
                symbol=symbol, type="limit", side=side, amount=quantity, price=tp_price
            )
            logger.info(f"[Execution] TP {side.upper()} {quantity} {symbol} @ {tp_price} → {order['id']}")
            return order["id"]
        except Exception as e:
            logger.error(f"[Execution] Failed to place TP for {symbol}: {e}")
        return None
