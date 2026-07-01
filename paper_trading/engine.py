# ============================================================================
# PETROQUANT PAPER TRADING — MAIN ENGINE
# ============================================================================
# Orchestrates the full paper trading loop. Supports all timeframes via config.
#
# Every loop_interval seconds:
#   1. Fetch latest OHLCV candles (yfinance) at the configured interval
#   2. Build features (EMA, RSI, VWAP, ATR, etc.) — timeframe-aware
#   3. Retrain XGBoost if overdue
#   4. Generate signal: BUY / SELL / HOLD
#   5. Check market hours (skip if maintenance window)
#   6. Get daily regime (HMM, cached per day)
#   7. Execute via OrderEngine (with slippage + commission)
#   8. Log everything to SQLite
#   9. Regenerate HTML dashboard
#   10. Sleep until next bar
#
# Bug Fixes:
#   #2 — Loop log message now shows actual self.loop_interval (not hardcoded 60)
#   #4 — reset_account() now updates self.dashboard.portfolio so the dashboard
#         always shows fresh state after a reset
#
# PaperTradingEngine:
#   start()              — begins the trading loop
#   stop()               — gracefully stops
#   run_once()           — execute one cycle (for testing)
#   reset_account()      — wipe trades and reset portfolio
#   switch_timeframe(tf) — change timeframe at runtime without restarting process
#   uptime_seconds()     — how long the engine has been running
# ============================================================================

import time
import threading
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Setup logging before imports ──────────────────────────────────────────────
def _setup_logging(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )

from . import config as cfg

_setup_logging(cfg.LOG_PATH)
logger = logging.getLogger(__name__)

from .price_feed        import fetch_candles, fetch_latest_price, is_market_open, get_market_status
from .features_intraday import build_features, build_target
from .model_intraday    import IntradaySignalModel
from .daily_regime      import DailyRegimeDetector
from .portfolio         import Portfolio
from .order_engine      import OrderEngine
from .trade_log         import TradeLog
from .dashboard_live    import LiveDashboard


# Helper: map yfinance interval string to minutes
_INTERVAL_TO_MINUTES = {
    '1m': 1, '5m': 5, '15m': 15, '30m': 30,
    '1h': 60, '2h': 120, '4h': 240, '1d': 1440,
}


class PaperTradingEngine:
    """
    Orchestrates the PetroQuant paper trading loop.

    Parameters
    ----------
    timeframe : str — '1m', '5m', '15m', '1h', or '1d'
                      Applies config.TIMEFRAME_CONFIGS settings for this timeframe.
    """

    def __init__(self, timeframe: str = '1m'):
        # Apply the requested timeframe — sets all config module-level vars
        cfg.apply_timeframe(timeframe)

        self.loop_interval = cfg.LOOP_INTERVAL_SECS
        self._running      = False
        self._thread       = None
        self._start_time   = None

        # ── Core components ──────────────────────────────────────────────────
        self.trade_log    = TradeLog(cfg.DB_PATH)
        self.portfolio    = Portfolio()
        self.model        = IntradaySignalModel()
        self.regime_det   = DailyRegimeDetector()
        self.order_engine = OrderEngine(self.portfolio, self.trade_log)

        # ── Restore state from DB (survives server restarts) ─────────────────
        restored = self.portfolio.restore_from_db(self.trade_log)
        if not restored:
            logger.info("[Engine] Fresh start — no prior trade history found in DB")

        # ── Live state (read by web server) ──────────────────────────────────
        self.last_price          = None
        self.last_signal         = 'HOLD'
        self.last_prob           = 0.5
        self.current_regime      = 'CHOPPY'
        self.current_regime_mult = 0.5
        self._last_df            = None   # latest OHLCV for dashboard

        # ── Dashboard ─────────────────────────────────────────────────────────
        self.dashboard = LiveDashboard(
            trade_log  = self.trade_log,
            portfolio  = self.portfolio,
        )

        logger.info("=" * 60)
        logger.info("  PetroQuant Paper Trading Engine — Initialized")
        logger.info(f"  Capital   : ${cfg.INITIAL_CAPITAL:,.2f}")
        logger.info(f"  Ticker    : {cfg.TICKER_INTRADAY} ({cfg.CANDLE_INTERVAL} bars)")
        logger.info(f"  Timeframe : {cfg.ACTIVE_TIMEFRAME} "
                    f"({cfg.get_timeframe_label()})")
        logger.info(f"  Horizon   : {cfg.PREDICT_HORIZON} bars")
        logger.info(f"  Loop      : every {cfg.LOOP_INTERVAL_SECS}s")
        logger.info(f"  DB        : {cfg.DB_PATH}")
        logger.info("=" * 60)

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self):
        """Start the paper trading loop in a background thread."""
        if self._running:
            logger.warning("[Engine] Already running.")
            return
        self._running    = True
        self._start_time = datetime.utcnow()
        self._thread = threading.Thread(target=self._loop,
                                        name='TradingLoop', daemon=True)
        self._thread.start()
        logger.info("[Engine] Trading loop started.")

    def stop(self):
        """Gracefully stop the trading loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=15)
        logger.info("[Engine] Trading loop stopped.")

    def run_once(self) -> dict:
        """Execute exactly one trading cycle. Used for testing and manual runs."""
        return self._cycle()

    def is_market_open(self) -> bool:
        return is_market_open()

    def uptime_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return (datetime.utcnow() - self._start_time).total_seconds()

    def reset_account(self):
        """
        Wipe all trades and reset portfolio to initial capital.

        Bug #4 fix: also updates self.dashboard.portfolio so the dashboard
        always reflects the fresh portfolio object, not the old one.
        """
        with self.trade_log._get_conn() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM snapshots")
        self.portfolio    = Portfolio()
        self.order_engine = OrderEngine(self.portfolio, self.trade_log)
        self.model        = IntradaySignalModel()

        # Bug #4 fix: update dashboard reference to new portfolio
        self.dashboard.portfolio = self.portfolio

        logger.warning("[Engine] Account RESET — all trades cleared, portfolio restarted.")

    def switch_timeframe(self, timeframe: str) -> dict:
        """
        Switch the engine to a different trading timeframe at runtime.

        Stops the loop, applies the new config, resets the model
        (so it retrains on the new bar size), then restarts the loop.

        Returns dict with new settings for the API caller.
        """
        if timeframe not in cfg.TIMEFRAME_CONFIGS:
            raise ValueError(f"Unknown timeframe '{timeframe}'. "
                             f"Valid: {list(cfg.TIMEFRAME_CONFIGS.keys())}")

        logger.info(f"[Engine] Switching timeframe: {cfg.ACTIVE_TIMEFRAME} → {timeframe}")

        was_running = self._running
        if was_running:
            self.stop()

        # Apply new timeframe settings
        cfg.apply_timeframe(timeframe)
        self.loop_interval = cfg.LOOP_INTERVAL_SECS

        # Reset model so it retrains on first cycle with new bar size
        self.model = IntradaySignalModel()
        self.model.refresh_from_config()

        logger.info(f"[Engine] Timeframe switched to {timeframe} "
                    f"(loop={self.loop_interval}s, horizon={cfg.PREDICT_HORIZON}bars)")

        if was_running:
            self.start()

        return {
            'timeframe'   : cfg.ACTIVE_TIMEFRAME,
            'loop_secs'   : cfg.LOOP_INTERVAL_SECS,
            'horizon'     : cfg.PREDICT_HORIZON,
            'min_train'   : cfg.MIN_TRAIN_BARS,
            'retrain_mins': cfg.RETRAIN_EVERY_MINS,
        }

    # ── Trading Loop ──────────────────────────────────────────────────────────
    def _loop(self):
        """Main trading loop — runs forever until stop() is called."""
        # Bug #2 fix: log actual self.loop_interval, not hardcoded 60
        logger.info(f"[Engine] Loop started. Running every {self.loop_interval}s "
                    f"(timeframe: {cfg.ACTIVE_TIMEFRAME})")

        while self._running:
            cycle_start = time.time()

            try:
                self._cycle()
            except Exception as e:
                logger.error(f"[Engine] Unhandled error in cycle: {e}", exc_info=True)

            elapsed   = time.time() - cycle_start
            sleep_for = max(0, self.loop_interval - elapsed)
            logger.debug(f"[Engine] Cycle took {elapsed:.1f}s — sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)

    # ── Single Cycle ──────────────────────────────────────────────────────────
    def _cycle(self) -> dict:
        """
        One complete trading cycle:
          fetch → features → signal → regime → execute → log → dashboard
        """
        now = datetime.utcnow()
        logger.info(f"[Engine] ---- Cycle @ {now.strftime('%Y-%m-%d %H:%M:%S UTC')} "
                    f"| TF={cfg.ACTIVE_TIMEFRAME} ----")

        # ── Step 1: Check market hours ────────────────────────────────────────
        market_status = get_market_status()
        if not market_status['is_open']:
            logger.info(f"[Engine] Market CLOSED ({market_status['note']}) — skipping")
            return {'status': 'market_closed', 'reason': market_status['note']}

        # ── Step 2: Fetch candles ─────────────────────────────────────────────
        logger.info(f"[Engine] Fetching {cfg.CANDLE_INTERVAL} WTI candles...")
        df_raw = fetch_candles(
            interval=cfg.CANDLE_INTERVAL,
            days=cfg.BARS_TO_FETCH,
        )
        if df_raw.empty:
            logger.warning("[Engine] No data from yfinance — skipping cycle")
            return {'status': 'no_data'}

        self._last_df   = df_raw
        self.last_price = float(df_raw['Close'].iloc[-1])
        logger.info(f"[Engine] Fetched {len(df_raw)} bars | WTI: ${self.last_price:.4f}")

        # ── Step 3: Get daily regime (cached per day) ─────────────────────────
        regime, regime_mult = self.regime_det.get_regime()
        self.current_regime      = regime
        self.current_regime_mult = regime_mult
        logger.info(f"[Engine] Regime: {regime} (mult: {regime_mult})")

        # ── Step 4: Build features ────────────────────────────────────────────
        tf_minutes = _INTERVAL_TO_MINUTES.get(cfg.CANDLE_INTERVAL, 1)
        feat_df = build_features(df_raw, regime=regime,
                                 timeframe_minutes=tf_minutes)
        if feat_df.empty:
            logger.warning("[Engine] Feature build failed — skipping cycle")
            return {'status': 'feature_error'}

        # ── Step 5: Train or retrain model ────────────────────────────────────
        if self.model.should_retrain():
            logger.info("[Engine] Retraining XGBoost model...")
            labeled_df   = build_target(feat_df, horizon=cfg.PREDICT_HORIZON)
            train_result = self.model.train(labeled_df)
            logger.info(f"[Engine] Retrain result: {train_result}")

        # ── Step 6: Generate signal ───────────────────────────────────────────
        signal, prob = self.model.predict_latest(feat_df)
        self.last_signal = signal
        self.last_prob   = prob
        logger.info(f"[Engine] Signal: {signal} | Prob(UP): {prob:.2%}")

        # ── Step 7: Execute order ─────────────────────────────────────────────
        result = self.order_engine.execute(
            signal      = signal,
            probability = prob,
            price       = self.last_price,
            regime      = regime,
            regime_mult = regime_mult,
            bar_time    = now,
        )
        logger.info(f"[Engine] Execution: {result.get('action')} | "
                    f"equity=${result.get('equity', self.portfolio.cash):,.2f}")

        # ── Step 8: Regenerate dashboard ──────────────────────────────────────
        try:
            self.dashboard.price_feed_df      = df_raw
            self.dashboard.feature_importance = self.model.get_feature_importance()
            html = self.dashboard.render(
                current_price = self.last_price,
                regime        = regime,
                model_status  = self.model.get_model_status(),
            )
            self.dashboard.save(html)
        except Exception as e:
            logger.warning(f"[Engine] Dashboard render error: {e}")

        # ── Step 9: Log portfolio snapshot ────────────────────────────────────
        snap = self.portfolio.get_snapshot(self.last_price)
        logger.info(
            f"[Engine] Portfolio | equity=${snap['equity']:,.2f} | "
            f"return={snap['total_return_pct']:+.3f}% | "
            f"position={'LONG/SHORT' if snap['open_position'] else 'FLAT'} | "
            f"trades={snap['total_trades']}"
        )

        return {
            'status'     : 'ok',
            'signal'     : signal,
            'probability': prob,
            'price'      : self.last_price,
            'regime'     : regime,
            'action'     : result.get('action'),
            'equity'     : snap['equity'],
            'return_pct' : snap['total_return_pct'],
            'timeframe'  : cfg.ACTIVE_TIMEFRAME,
        }
