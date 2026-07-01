# ============================================================================
# PETROQUANT PAPER TRADING — PRICE FEED
# ============================================================================
# Fetches live WTI OHLCV candles from Yahoo Finance (yfinance >= 1.0).
#
# yfinance 1.x changed the API — ticker.history() returns MultiIndex columns
# like ('Close','CL=F') and may return None internally. We use yf.download()
# which is stable and flatten the MultiIndex automatically.
#
# Supports timeframes: 1m, 5m, 15m, 1h, 1d
# yfinance data limits (approximate):
#   1m  — last 7 calendar days
#   5m  — last 60 days
#   15m — last 60 days
#   1h  — last 730 days
#   1d  — unlimited
#
# Key functions:
#   fetch_candles()       — returns DataFrame of OHLCV bars (any interval)
#   fetch_latest_price()  — returns single latest close price (float)
#   is_market_open()      — True if WTI futures market is currently active
#   get_market_status()   — dict with status details
# ============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import time

from . import config as cfg

logger = logging.getLogger(__name__)

# Eastern time zone (WTI market hours reference)
ET  = pytz.timezone("US/Eastern")
UTC = pytz.utc


# ─────────────────────────────────────────────────────────────────────────────
def fetch_candles(ticker: str = None,
                  interval: str = None,
                  days: int = None,
                  retries: int = 3) -> pd.DataFrame:
    """
    Fetch OHLCV candles from yfinance for any supported interval.

    Uses yf.download() (stable across yfinance 0.x and 1.x) and flattens
    MultiIndex columns that yfinance 1.x returns.

    Parameters
    ----------
    ticker   : str  — Yahoo Finance ticker (default from config: 'CL=F')
    interval : str  — candle interval ('1m','5m','15m','1h','1d')
    days     : int  — days of history to fetch
    retries  : int  — retry attempts on failure

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
                 DatetimeIndex in UTC, tz-naive
    """
    ticker   = ticker   or cfg.TICKER_INTRADAY
    interval = interval or cfg.CANDLE_INTERVAL
    days     = days     or cfg.BARS_TO_FETCH

    for attempt in range(retries):
        try:
            df = yf.download(
                tickers      = ticker,
                period       = f"{days}d",
                interval     = interval,
                auto_adjust  = True,
                progress     = False,
                prepost      = True,    # Include pre/post market (WTI ≈ 24/7)
            )

            if df is None or df.empty:
                logger.warning(f"[PriceFeed] Empty response from yfinance "
                               f"(attempt {attempt+1}, interval={interval})")
                time.sleep(2 ** attempt)
                continue

            # ── Flatten MultiIndex columns (yfinance 1.x returns them) ───────
            # e.g. ('Close','CL=F') → 'Close'
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # ── Normalise index to tz-naive UTC ──────────────────────────────
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(UTC).tz_localize(None)

            # ── Remove duplicates & sort ──────────────────────────────────────
            df = df[~df.index.duplicated(keep='last')].sort_index()

            # ── Keep only OHLCV columns ──────────────────────────────────────
            ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume']
                          if c in df.columns]
            if not ohlcv_cols:
                logger.warning(f"[PriceFeed] No OHLCV columns found. Got: {df.columns.tolist()}")
                time.sleep(2 ** attempt)
                continue
            df = df[ohlcv_cols].copy()

            # ── Drop zero/null rows ──────────────────────────────────────────
            df = df.dropna(subset=['Close'])
            df = df[df['Close'] > 0]

            # ── Drop maintenance-hour gaps (5PM-6PM ET) — intraday only ──────
            if interval in ('1m', '5m', '15m', '1h'):
                df = _drop_maintenance_hour(df)

            logger.info(f"[PriceFeed] Fetched {len(df)} {interval} bars "
                        f"({df.index[0]} -> {df.index[-1]}) | "
                        f"Latest close: ${df['Close'].iloc[-1]:.4f}")
            return df

        except Exception as e:
            logger.error(f"[PriceFeed] Error on attempt {attempt+1}: {e}", exc_info=False)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    logger.error("[PriceFeed] All retry attempts failed. Returning empty DataFrame.")
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


# Backward-compatible alias (used in --dashboard mode)
def fetch_1min_candles(ticker: str = None, days: int = None,
                       retries: int = 3) -> pd.DataFrame:
    """Backward-compatible alias for fetch_candles() at '1m' interval."""
    return fetch_candles(ticker=ticker, interval='1m', days=days or 7,
                         retries=retries)


# ─────────────────────────────────────────────────────────────────────────────
def fetch_latest_price(ticker: str = None) -> float | None:
    """
    Fetch just the latest close price for marking positions to market.

    Returns
    -------
    float — latest close price, or None on failure
    """
    ticker = ticker or cfg.TICKER_INTRADAY
    try:
        df = fetch_candles(ticker=ticker, interval='1m', days=2)
        if not df.empty:
            return float(df['Close'].iloc[-1])

        # Fallback: fast_info
        info  = yf.Ticker(ticker).fast_info
        price = getattr(info, 'last_price', None)
        return float(price) if price else None
    except Exception as e:
        logger.error(f"[PriceFeed] fetch_latest_price error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
def is_market_open(ticker: str = None) -> bool:
    """
    Check if WTI futures market is currently active.

    WTI (CL=F) trades nearly 24/7:
      Sun 6:00 PM ET → Fri 5:00 PM ET
      Daily maintenance: 5:00 PM - 6:00 PM ET
    """
    now_et  = datetime.now(ET)
    weekday = now_et.weekday()
    hour    = now_et.hour

    if weekday == 5:              # Saturday — all day closed
        return False
    if weekday == 6 and hour < 18: # Sunday before 6PM ET
        return False
    if weekday == 4 and hour >= 17: # Friday after 5PM ET
        return False
    if hour == 17:                 # Daily maintenance 5-6PM ET
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
def get_market_status() -> dict:
    """Returns a dict describing current market status."""
    now_et  = datetime.now(ET)
    is_open = is_market_open()

    return {
        'is_open'    : is_open,
        'status'     : 'OPEN' if is_open else 'CLOSED',
        'current_et' : now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
        'weekday'    : now_et.strftime('%A'),
        'note'       : 'WTI maintenance 5-6 PM ET' if now_et.hour == 17 else
                       ('Weekend' if now_et.weekday() >= 5 else 'Normal session'),
        'timeframe'  : cfg.ACTIVE_TIMEFRAME,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _drop_maintenance_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows in the WTI maintenance window (5:00-6:00 PM ET).
    Guards against both tz-naive and tz-aware indexes.
    """
    if df.empty:
        return df

    if df.index.tz is None:
        df_et_index = df.index.tz_localize(UTC).tz_convert(ET)
    else:
        df_et_index = df.index.tz_convert(ET)

    return df[df_et_index.hour != 17]
