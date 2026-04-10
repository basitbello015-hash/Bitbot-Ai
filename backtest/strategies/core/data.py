"""
core/data.py
Market Data Module — fetches OHLCV candles and current prices from Bybit via ccxt.
"""

import logging
import time
from typing import Optional
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class MarketData:
    TIMEFRAME_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "15m",
        limit: int = 200, retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe '{timeframe}'.")

        for attempt in range(1, retries + 1):
            try:
                raw = self.exchange.fetch_ohlcv(
                    symbol, timeframe=self.TIMEFRAME_MAP[timeframe], limit=limit
                )
                if not raw:
                    return None
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
                df.set_index("timestamp", inplace=True)
                return df
            except ccxt.NetworkError as e:
                logger.warning(f"[{symbol}] Network error (attempt {attempt}/{retries}): {e}")
                time.sleep(2 ** attempt)
            except ccxt.ExchangeError as e:
                logger.error(f"[{symbol}] Exchange error: {e}")
                return None

        logger.error(f"[{symbol}] All {retries} fetch attempts failed.")
        return None

    def fetch_current_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            return float(price) if price else None
        except Exception as e:
            logger.error(f"[{symbol}] fetch_current_price error: {e}")
            return None

    def fetch_balance(self) -> dict:
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"fetch_balance error: {e}")
            return {}

    def get_usdt_balance(self) -> float:
        balance = self.fetch_balance()
        try:
            return float(balance.get("free", {}).get("USDT", 0.0))
        except Exception:
            return 0.0
