# ============================================================================
# PETROQUANT PAPER TRADING — CONFIGURATION
# ============================================================================
# All tunable parameters in one place.
# Edit this file to change account size, thresholds, or market settings.
# ============================================================================

import os

# ── Account Settings ─────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 1_000_000      # Starting paper account in USD
CURRENCY           = "USD"          # Display currency

# ── Market / Price Feed ──────────────────────────────────────────────────────
TICKER_INTRADAY    = "CL=F"         # WTI Crude Oil Futures (1-min)
TICKER_DAILY       = "CL=F"         # WTI Crude Oil Futures (daily, for HMM)
CANDLE_INTERVAL    = "1m"           # Candle size: '1m' = 1 minute
BARS_TO_FETCH      = 7              # yfinance 1-min limit: last 7 days (in days)
BARS_FOR_TRAINING  = 5000           # approx 3.5 days of 1-min bars

# ── Signal / Prediction ──────────────────────────────────────────────────────
PREDICT_HORIZON    = 5              # Predict price direction in N minutes
BUY_THRESHOLD      = 0.52           # Model prob > 52% → BUY  (loosened from 58%)
SELL_THRESHOLD     = 0.48           # Model prob < 48% → SELL (loosened from 42%)
                                    # 48%-52% → HOLD (tight band to avoid noise)

# ── Model Retraining ─────────────────────────────────────────────────────────
RETRAIN_EVERY_MINS = 240            # Retrain XGBoost every 4 hours
MIN_TRAIN_BARS     = 500            # Minimum bars needed before trading

# ── Execution / Risk ─────────────────────────────────────────────────────────
COMMISSION_PCT     = 0.0002         # 2 basis points per side (round-trip = 4 bps)
SLIPPAGE_PCT       = 0.0001         # 1 bp slippage per trade
MAX_POSITION_PCT   = 0.15           # Max 15% of equity in a single trade
MAX_DAILY_LOSS_PCT = 0.05           # Stop trading if daily loss > 5% of equity

# ── Regime-Aware Position Sizing (from daily HMM) ────────────────────────────
REGIME_MULTIPLIERS = {
    'BULL':   1.00,
    'CHOPPY': 0.60,   # raised from 0.50
    'PANIC':  0.40,   # raised from 0.25 — still cautious but allows trades
}

# ── WTI Market Hours (UTC) ───────────────────────────────────────────────────
# WTI Futures trades Sun-Fri, 23 hrs/day
# Maintenance gap: 5:00 PM - 6:00 PM ET = 21:00-22:00 UTC (winter) / 22:00-23:00 UTC (summer)
# We use ET-based logic for simplicity
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
