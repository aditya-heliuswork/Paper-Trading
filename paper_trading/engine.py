# ============================================================================
# PETROQUANT PAPER TRADING — MAIN ENGINE
# ============================================================================
# Orchestrates the full paper trading loop:
#   Every 60 seconds:
#     1. Fetch latest 1-min WTI candles (yfinance)
#     2. Build intraday features (EMA, RSI, VWAP, ATR, etc.)
#     3. Retrain XGBoost if overdue (every 4 hours)
#     4. Generate signal: BUY / SELL / HOLD
#     5. Check market hours (skip if maintenance window)
#     6. Get daily regime (HMM, cached per day)
#     7. Execute via OrderEngine (with slippage + commission)
#     8. Log everything to SQLite
#     9. Regenerate HTML dashboard
#    10. Sleep until next bar
#
# PaperTradingEngine:
#   start()           — begins the trading loop
#   stop()            — gracefully stops
#   run_once()        — execute one cycle (for testing)
#   reset_account()   — wipe trades and reset portfolio
#   uptime_seconds()  — how long the engine has been running
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

from .price_feed       import fetch_1min_candles, fetch_latest_price, is_market_open, get_market_status
from .features_intraday import build_features, build_target
from .model_intraday   import IntradaySignalModel
from .daily_regime     import DailyRegimeDetector
from .portfolio        import Portfolio
from .order_engine     import OrderEngine
from .trade_log        import TradeLog
from .dashboard_live   import LiveDashboard


class PaperTradingEngine:
    """
    Orchestrates the PetroQuant paper trading loop.

    Parameters
    ----------
    loop_interval_secs : int  — seconds between each trading cycle (default 60 = 1 bar)
    """

    def __init__(self, loop_interval_secs: int = 60):
        self.loop_interval    = loop_interval_secs
        self._running         = False
        self._thread          = None
        self._start_time      = None

        # ── Core components ──────────────────────────────────────────────────
        self.trade_log   = TradeLog(cfg.DB_PATH)
        self.portfolio   = Portfolio()
        self.model       = IntradaySignalModel()
        self.regime_det  = DailyRegimeDetector()
        self.order_engine= OrderEngine(self.portfolio, self.trade_log)

        # ── Restore state from DB (survives server restarts) ─────────────────
        restored = self.portfolio.restore_from_db(self.trade_log)
        if not restored:
            logger.info("[Engine] Fresh start — no prior trade history found in DB")

        # ── Live state (read by web server) ──────────────────────────────────
        self.last_price      = None
        self.last_signal     = 'HOLD'
        self.last_prob       = 0.5
        self.current_regime  = 'CHOPPY'
        self.current_regime_mult = 0.5
        self._last_df        = None     # latest 1-min OHLCV for dashboard

        # ── Dashboard ─────────────────────────────────────────────────────────
        self.dashboard = LiveDashboard(
            trade_log  = self.trade_log,
            portfolio  = self.portfolio,
        )

        logger.info("=" * 60)
        logger.info("  PetroQuant Paper Trading Engine — Initialized")
        logger.info(f"  Capital  : ${cfg.INITIAL_CAPITAL:,.2f}")
        logger.info(f"  Ticker   : {cfg.TICKER_INTRADAY} ({cfg.CANDLE_INTERVAL} bars)")
        logger.info(f"  Horizon  : {cfg.PREDICT_HORIZON} minutes")
        logger.info(f"  DB       : {cfg.DB_PATH}")
        logger.info("=" * 60)

    # ── Public API ────────────────────────────────────────────────────────────
    def start(self):
        """Start the paper trading loop in a background thread."""
        if self._running:
            logger.warning("[Engine] Already running.")
            return
        self._running    = True
        self._start_time = datetime.utcnow()
        self._thread = threading.Thread(target=self._loop, name='TradingLoop', daemon=True)
        self._thread.start()
        logger.info("[Engine] Trading loop started.")

    def stop(self):
        """Gracefully stop the trading loop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
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
        """Wipe all trades and reset portfolio to initial capital."""
        with self.trade_log._get_conn() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM snapshots")
        self.portfolio    = Portfolio()
        self.order_engine = OrderEngine(self.portfolio, self.trade_log)
        self.model        = IntradaySignalModel()
        logger.warning("[Engine] Account RESET — all trades cleared, portfolio restarted.")

    # ── Trading Loop ──────────────────────────────────────────────────────────
    def _loop(self):
        """Main trading loop — runs forever until stop() is called."""
        logger.info("[Engine] Loop started. Running every 60 seconds...")

        while self._running:
            cycle_start = time.time()

            try:
                self._cycle()
            except Exception as e:
                logger.error(f"[Engine] Unhandled error in cycle: {e}", exc_info=True)

            # Sleep until next bar (accounting for cycle execution time)
            elapsed = time.time() - cycle_start
            sleep_for = max(0, self.loop_interval - elapsed)
            logger.debug(f"[Engine] Cycle took {elapsed:.1f}s — sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)

    # ── Single Cycle ──────────────────────────────────────────────────────────
    def _cycle(self) -> dict:
        """
        One complete trading cycle:
          fetch -> features -> signal -> regime -> execute -> log -> dashboard
        """
        now = datetime.utcnow()
        logger.info(f"[Engine] ---- Cycle @ {now.strftime('%Y-%m-%d %H:%M:%S UTC')} ----")

        # ── Step 1: Check market hours ────────────────────────────────────────
        market_status = get_market_status()
        if not market_status['is_open']:
            logger.info(f"[Engine] Market CLOSED ({market_status['note']}) — skipping cycle")
            return {'status': 'market_closed', 'reason': market_status['note']}

        # ── Step 2: Fetch 1-min candles ───────────────────────────────────────
        logger.info("[Engine] Fetching 1-min WTI candles...")
        df_raw = fetch_1min_candles()
        if df_raw.empty:
            logger.warning("[Engine] No data from yfinance — skipping cycle")
            return {'status': 'no_data'}

        self._last_df    = df_raw
        self.last_price  = float(df_raw['Close'].iloc[-1])
        logger.info(f"[Engine] Fetched {len(df_raw)} bars | Latest WTI: ${self.last_price:.4f}")

        # ── Step 3: Get daily regime (cached per day) ─────────────────────────
        regime, regime_mult = self.regime_det.get_regime()
        self.current_regime      = regime
        self.current_regime_mult = regime_mult
        logger.info(f"[Engine] Regime: {regime} (position mult: {regime_mult})")

        # ── Step 4: Build features ────────────────────────────────────────────
        feat_df = build_features(df_raw, regime=regime)
        if feat_df.empty:
            logger.warning("[Engine] Feature build failed — skipping cycle")
            return {'status': 'feature_error'}

        # ── Step 5: Train or retrain model ────────────────────────────────────
        if self.model.should_retrain():
            logger.info("[Engine] Retraining XGBoost model...")
            labeled_df = build_target(feat_df, horizon=cfg.PREDICT_HORIZON)
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
            self.dashboard.price_feed_df     = df_raw
            self.dashboard.feature_importance= self.model.get_feature_importance()
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
            f"position={'LONG' if snap['open_position'] else 'FLAT'} | "
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
        }
