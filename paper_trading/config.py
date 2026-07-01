# ============================================================================
# PETROQUANT PAPER TRADING — CONFIGURATION
# ============================================================================
# All tunable parameters in one place.
# 1-minute trading has been REMOVED. Minimum timeframe is 5 minutes.
#
# Supported timeframes: 5m, 15m, 1h, 1d
# Default: 5m (runs every 5 minutes, predicts 15 minutes ahead)
#
# call apply_timeframe("15m") to switch at runtime.
# ============================================================================

import os

# ── Account Settings ─────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 1_000_000      # Starting paper account in USD
CURRENCY           = "USD"

# ── Signal / Prediction ──────────────────────────────────────────────────────
BUY_THRESHOLD      = 0.52           # Model prob > 52% -> BUY
SELL_THRESHOLD     = 0.48           # Model prob < 48% -> SELL

# ── Execution / Risk ─────────────────────────────────────────────────────────
COMMISSION_PCT     = 0.0002         # 2 basis points per side
SLIPPAGE_PCT       = 0.0001         # 1 bp slippage per trade
MAX_POSITION_PCT   = 0.15           # Max 15% of equity in a single trade
MAX_DAILY_LOSS_PCT = 0.05           # Stop trading if daily loss > 5%

# ── Regime-Aware Position Sizing ─────────────────────────────────────────────
REGIME_MULTIPLIERS = {
    'BULL':   1.00,
    'CHOPPY': 0.60,
    'PANIC':  0.40,
}

# ── WTI Ticker ───────────────────────────────────────────────────────────────
TICKER_INTRADAY    = "CL=F"
TICKER_DAILY       = "CL=F"

# ── Active Defaults (overridden by apply_timeframe) ───────────────────────────
CANDLE_INTERVAL    = "5m"           # Default: 5-minute candles
BARS_TO_FETCH      = 30
BARS_FOR_TRAINING  = 2000
PREDICT_HORIZON    = 3              # 3 bars x 5m = 15 min ahead
LOOP_INTERVAL_SECS = 300            # Run every 5 minutes
MIN_TRAIN_BARS     = 150
RETRAIN_EVERY_MINS = 480

# ── WTI Market Hours ─────────────────────────────────────────────────────────
MAINTENANCE_START  = "17:00"
MAINTENANCE_END    = "18:00"

# ── Storage ──────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR         = os.path.join(BASE_DIR, 'output')
DB_PATH            = os.path.join(OUTPUT_DIR, 'paper_trades.db')
LOG_PATH           = os.path.join(OUTPUT_DIR, 'paper_trading.log')
DASHBOARD_PATH     = os.path.join(OUTPUT_DIR, 'paper_dashboard.html')

# ── Web Server ───────────────────────────────────────────────────────────────
WEB_PORT           = int(os.environ.get('PORT', 8080))
DASHBOARD_REFRESH  = 60

# =============================================================================
# TIMEFRAME CONFIGURATIONS
# 1m has been REMOVED. Use 5m as the fastest timeframe.
# =============================================================================
#
#  5m  — runs every 5 min  — predicts 15 min ahead  — great for intraday
# 15m  — runs every 15 min — predicts 30 min ahead  — less noise, more reliable
#  1h  — runs every 1 hr   — predicts 2 hrs ahead   — swing-style
#  1d  — runs once/day     — predicts next day dir   — position trading

TIMEFRAME_CONFIGS = {
    "5m": {
        "interval"          : "5m",
        "bars_to_fetch_days": 30,
        "loop_secs"         : 300,
        "predict_horizon"   : 3,       # 3 x 5m = 15 min ahead
        "min_train_bars"    : 150,
        "retrain_mins"      : 480,     # retrain every 8 hours
        "bars_for_training" : 2000,
        "label"             : "5 Min",
        "description"       : "5 Minutes — trade every 5 min, predict 15 min ahead",
    },
    "15m": {
        "interval"          : "15m",
        "bars_to_fetch_days": 60,
        "loop_secs"         : 900,
        "predict_horizon"   : 2,       # 2 x 15m = 30 min ahead
        "min_train_bars"    : 100,
        "retrain_mins"      : 720,     # retrain every 12 hours
        "bars_for_training" : 1000,
        "label"             : "15 Min",
        "description"       : "15 Minutes — trade every 15 min, predict 30 min ahead",
    },
    "1h": {
        "interval"          : "1h",
        "bars_to_fetch_days": 180,
        "loop_secs"         : 3600,
        "predict_horizon"   : 2,       # 2 x 1h = 2 hours ahead
        "min_train_bars"    : 80,
        "retrain_mins"      : 1440,    # retrain once per day
        "bars_for_training" : 500,
        "label"             : "1 Hour",
        "description"       : "1 Hour — trade every hour, predict 2 hours ahead",
    },
    "1d": {
        "interval"          : "1d",
        "bars_to_fetch_days": 730,
        "loop_secs"         : 86400,
        "predict_horizon"   : 1,       # predict next day
        "min_train_bars"    : 80,
        "retrain_mins"      : 10080,   # retrain weekly
        "bars_for_training" : 400,
        "label"             : "1 Day",
        "description"       : "1 Day — trade once daily, predict tomorrow's direction",
    },
}

# Active timeframe (set by apply_timeframe)
ACTIVE_TIMEFRAME = "5m"


def apply_timeframe(tf: str) -> None:
    """
    Switch the active trading timeframe at runtime.
    Raises ValueError if tf is not one of: 5m, 15m, 1h, 1d
    """
    if tf not in TIMEFRAME_CONFIGS:
        raise ValueError(
            f"Unknown timeframe '{tf}'. Valid options: {list(TIMEFRAME_CONFIGS.keys())}"
        )

    global ACTIVE_TIMEFRAME, CANDLE_INTERVAL, BARS_TO_FETCH, BARS_FOR_TRAINING
    global PREDICT_HORIZON, LOOP_INTERVAL_SECS, MIN_TRAIN_BARS, RETRAIN_EVERY_MINS

    c = TIMEFRAME_CONFIGS[tf]
    ACTIVE_TIMEFRAME    = tf
    CANDLE_INTERVAL     = c["interval"]
    BARS_TO_FETCH       = c["bars_to_fetch_days"]
    BARS_FOR_TRAINING   = c["bars_for_training"]
    PREDICT_HORIZON     = c["predict_horizon"]
    LOOP_INTERVAL_SECS  = c["loop_secs"]
    MIN_TRAIN_BARS      = c["min_train_bars"]
    RETRAIN_EVERY_MINS  = c["retrain_mins"]


def get_timeframe_label(tf: str = None) -> str:
    """Returns the short label for a timeframe."""
    tf = tf or ACTIVE_TIMEFRAME
    return TIMEFRAME_CONFIGS.get(tf, {}).get("label", tf)


def get_timeframe_description(tf: str = None) -> str:
    """Returns full description for a timeframe."""
    tf = tf or ACTIVE_TIMEFRAME
    return TIMEFRAME_CONFIGS.get(tf, {}).get("description", tf)
