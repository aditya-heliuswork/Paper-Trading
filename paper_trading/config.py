# ============================================================================
# PETROQUANT PAPER TRADING — CONFIGURATION
# ============================================================================
# All tunable parameters in one place.
# Edit this file to change account size, thresholds, or market settings.
#
# Multi-timeframe support:
#   call apply_timeframe("5m") to switch the active timeframe at runtime.
#   This updates all module-level variables (CANDLE_INTERVAL, BARS_TO_FETCH, etc.)
# ============================================================================

import os

# ── Account Settings ─────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 1_000_000      # Starting paper account in USD
CURRENCY           = "USD"          # Display currency

# ── Signal / Prediction ──────────────────────────────────────────────────────
BUY_THRESHOLD      = 0.52           # Model prob > 52% → BUY
SELL_THRESHOLD     = 0.48           # Model prob < 48% → SELL
                                    # 48%-52% → HOLD (tight band to avoid noise)

# ── Execution / Risk ─────────────────────────────────────────────────────────
COMMISSION_PCT     = 0.0002         # 2 basis points per side (round-trip = 4 bps)
SLIPPAGE_PCT       = 0.0001         # 1 bp slippage per trade
MAX_POSITION_PCT   = 0.15           # Max 15% of equity in a single trade
MAX_DAILY_LOSS_PCT = 0.05           # Stop trading if daily loss > 5% of equity

# ── Regime-Aware Position Sizing (from daily HMM) ────────────────────────────
REGIME_MULTIPLIERS = {
    'BULL':   1.00,
    'CHOPPY': 0.60,
    'PANIC':  0.40,
}

# ── WTI Market / Price Feed — Active Defaults (overridden by apply_timeframe) ─
TICKER_INTRADAY    = "CL=F"         # WTI Crude Oil Futures
TICKER_DAILY       = "CL=F"         # WTI Crude Oil Futures (daily, for HMM)
CANDLE_INTERVAL    = "1m"           # Active candle interval
BARS_TO_FETCH      = 7              # Days of history to fetch
BARS_FOR_TRAINING  = 5000           # Approx bars for feature engineering
PREDICT_HORIZON    = 5              # Predict price direction N bars ahead
LOOP_INTERVAL_SECS = 60             # Seconds between trading cycles
MIN_TRAIN_BARS     = 500            # Minimum bars needed before trading
RETRAIN_EVERY_MINS = 240            # Retrain XGBoost every N minutes

# ── WTI Market Hours (UTC) ───────────────────────────────────────────────────
MARKET_OPEN_ET     = "18:00"        # 6:00 PM ET Sunday (start of week)
MARKET_CLOSE_ET    = "17:00"        # 5:00 PM ET (daily close / maintenance)
MAINTENANCE_START  = "17:00"        # Skip trades between 5:00-6:00 PM ET
MAINTENANCE_END    = "18:00"        # Resume at 6:00 PM ET

# ── Storage ──────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR         = os.path.join(BASE_DIR, 'output')
DB_PATH            = os.path.join(OUTPUT_DIR, 'paper_trades.db')
LOG_PATH           = os.path.join(OUTPUT_DIR, 'paper_trading.log')
DASHBOARD_PATH     = os.path.join(OUTPUT_DIR, 'paper_dashboard.html')

# ── Web Server ───────────────────────────────────────────────────────────────
WEB_PORT           = int(os.environ.get('PORT', 8080))  # Railway injects $PORT
DASHBOARD_REFRESH  = 60             # Dashboard auto-refresh interval in seconds

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Each timeframe entry contains the settings that change with the resolution.
#
# Fields:
#   interval          — yfinance interval string
#   bars_to_fetch_days— days of history to request
#   loop_secs         — seconds between trading cycles
#   predict_horizon   — N bars ahead to predict direction
#   min_train_bars    — minimum bars needed to train XGBoost
#   retrain_mins      — how often to retrain (in minutes)
#   label             — human-readable label for the dashboard
#   bars_for_training — approximate bar count for the training window

TIMEFRAME_CONFIGS = {
    "1m": {
        "interval"          : "1m",
        "bars_to_fetch_days": 7,
        "loop_secs"         : 60,
        "predict_horizon"   : 5,
        "min_train_bars"    : 500,
        "retrain_mins"      : 240,
        "bars_for_training" : 5000,
        "label"             : "1 Minute",
    },
    "5m": {
        "interval"          : "5m",
        "bars_to_fetch_days": 30,     # yfinance allows 60d for 5m
        "loop_secs"         : 300,
        "predict_horizon"   : 3,      # predict 3 × 5m = 15 minutes ahead
        "min_train_bars"    : 150,
        "retrain_mins"      : 480,    # retrain every 8 hours
        "bars_for_training" : 2000,
        "label"             : "5 Minutes",
    },
    "15m": {
        "interval"          : "15m",
        "bars_to_fetch_days": 60,
        "loop_secs"         : 900,
        "predict_horizon"   : 2,      # predict 2 × 15m = 30 minutes ahead
        "min_train_bars"    : 100,
        "retrain_mins"      : 720,    # retrain every 12 hours
        "bars_for_training" : 1000,
        "label"             : "15 Minutes",
    },
    "1h": {
        "interval"          : "1h",
        "bars_to_fetch_days": 180,    # yfinance allows 730d for hourly
        "loop_secs"         : 3600,
        "predict_horizon"   : 2,      # predict 2 × 1h = 2 hours ahead
        "min_train_bars"    : 80,
        "retrain_mins"      : 1440,   # retrain once per day
        "bars_for_training" : 500,
        "label"             : "1 Hour",
    },
    "1d": {
        "interval"          : "1d",
        "bars_to_fetch_days": 730,    # 2 years of daily bars
        "loop_secs"         : 86400,  # run once per day
        "predict_horizon"   : 1,      # predict next day direction
        "min_train_bars"    : 80,
        "retrain_mins"      : 10080,  # retrain weekly
        "bars_for_training" : 400,
        "label"             : "1 Day",
    },
}

# Active timeframe name (set by apply_timeframe)
ACTIVE_TIMEFRAME = "1m"


def apply_timeframe(tf: str) -> None:
    """
    Switch the active trading timeframe at runtime.

    Updates all module-level config variables (CANDLE_INTERVAL, LOOP_INTERVAL_SECS,
    PREDICT_HORIZON, MIN_TRAIN_BARS, RETRAIN_EVERY_MINS) to match the given timeframe.

    Parameters
    ----------
    tf : str — one of '1m', '5m', '15m', '1h', '1d'

    Raises
    ------
    ValueError if tf is not in TIMEFRAME_CONFIGS
    """
    if tf not in TIMEFRAME_CONFIGS:
        raise ValueError(
            f"Unknown timeframe '{tf}'. Valid options: {list(TIMEFRAME_CONFIGS.keys())}"
        )

    global ACTIVE_TIMEFRAME, CANDLE_INTERVAL, BARS_TO_FETCH, BARS_FOR_TRAINING
    global PREDICT_HORIZON, LOOP_INTERVAL_SECS, MIN_TRAIN_BARS, RETRAIN_EVERY_MINS

    cfg = TIMEFRAME_CONFIGS[tf]
    ACTIVE_TIMEFRAME    = tf
    CANDLE_INTERVAL     = cfg["interval"]
    BARS_TO_FETCH       = cfg["bars_to_fetch_days"]
    BARS_FOR_TRAINING   = cfg["bars_for_training"]
    PREDICT_HORIZON     = cfg["predict_horizon"]
    LOOP_INTERVAL_SECS  = cfg["loop_secs"]
    MIN_TRAIN_BARS      = cfg["min_train_bars"]
    RETRAIN_EVERY_MINS  = cfg["retrain_mins"]


def get_timeframe_label(tf: str = None) -> str:
    """Returns the human-readable label for a timeframe (or the active one)."""
    tf = tf or ACTIVE_TIMEFRAME
    return TIMEFRAME_CONFIGS.get(tf, {}).get("label", tf)
