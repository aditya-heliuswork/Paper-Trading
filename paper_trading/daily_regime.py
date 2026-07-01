# ============================================================================
# PETROQUANT PAPER TRADING — DAILY REGIME DETECTOR
# ============================================================================
# Fetches daily WTI data and runs the existing HMM regime detection.
# Used as a macro filter for the intraday paper trader:
#   BULL   regime → 100% position size
#   CHOPPY regime →  50% position size
#   PANIC  regime →  25% position size
#
# Result is cached for the current day (only re-fetches at market open).
#
# DailyRegimeDetector:
#   get_regime()    — returns ('BULL'/'CHOPPY'/'PANIC', multiplier)
#   should_refresh()— True if cache is stale (new day or market open)
# ============================================================================

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, date
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("[DailyRegime] hmmlearn not available — defaulting to CHOPPY regime")

from . import config as cfg


class DailyRegimeDetector:
    """
    Runs daily HMM regime detection on WTI price history.
    Provides a macro context filter for the intraday signal model.

    The result is cached per day — doesn't hit the API every minute.
    """

    def __init__(self, ticker: str = cfg.TICKER_DAILY, lookback_days: int = 365):
        self.ticker        = ticker
        self.lookback_days = lookback_days

        # Cache
        self._cached_regime = 'CHOPPY'   # default until first fetch
        self._cached_mult   = cfg.REGIME_MULTIPLIERS.get('CHOPPY', 0.5)
        self._cache_date    = None        # date of last successful fetch
        self._regime_stats  = {}

    # ── Public API ────────────────────────────────────────────────────────────
    def get_regime(self, force_refresh: bool = False) -> tuple[str, float]:
        """
        Returns current daily market regime and position size multiplier.

        Parameters
        ----------
        force_refresh : bool — bypass cache and refetch from yfinance

        Returns
        -------
        tuple: (regime_name, position_multiplier)
            regime_name : 'BULL' | 'CHOPPY' | 'PANIC'
            multiplier  : 1.0 / 0.5 / 0.25
        """
        if force_refresh or self.should_refresh():
            self._fetch_and_classify()

        return (self._cached_regime, self._cached_mult)

    def should_refresh(self) -> bool:
        """Returns True if the cache is stale (new UTC calendar day)."""
        if self._cache_date is None:
            return True
        return self._cache_date != datetime.utcnow().date()

    def get_regime_stats(self) -> dict:
        """Returns last computed regime statistics."""
        return self._regime_stats

    # ── Internal: fetch + HMM ─────────────────────────────────────────────────
    def _fetch_and_classify(self):
        """Fetches daily WTI data and runs HMM to classify current regime."""
        logger.info("[DailyRegime] Refreshing daily regime from yfinance...")

        try:
            # Fetch daily data
            tkr = yf.Ticker(self.ticker)
            df  = tkr.history(period=f'{self.lookback_days}d', interval='1d', auto_adjust=True)

            if df is None or df.empty or 'Close' not in df.columns:
                logger.warning("[DailyRegime] No daily data — keeping cached regime")
                return

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='last')].sort_index().dropna(subset=['Close'])

            if len(df) < 100:
                logger.warning(f"[DailyRegime] Only {len(df)} daily bars — not enough for HMM")
                return

            # ── Feature matrix for HMM (same as existing PetroQuant strategy) ──
            log_ret  = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            real_vol = log_ret.rolling(20).std() * np.sqrt(252)

            hmm_df = pd.DataFrame({
                'log_ret'  : log_ret,
                'real_vol' : real_vol,
            }).dropna()

            if len(hmm_df) < 50:
                return

            X = hmm_df.values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # ── Fit 3-state HMM ─────────────────────────────────────────────
            if not HMM_AVAILABLE:
                self._cached_regime = self._fallback_regime(df['Close'])
                self._cached_mult   = cfg.REGIME_MULTIPLIERS.get(self._cached_regime, 0.5)
                self._cache_date    = datetime.utcnow().date()
                return

            hmm = GaussianHMM(
                n_components=3, covariance_type='full',
                n_iter=200, random_state=42
            )
            hmm.fit(X_scaled)
            states = hmm.predict(X_scaled)

            # ── Label states by volatility ───────────────────────────────────
            state_vol  = {}
            state_ret  = {}
            state_cnt  = {}
            for s in range(3):
                mask = states == s
                if mask.sum() > 0:
                    state_vol[s] = hmm_df.loc[mask, 'real_vol'].mean()
                    state_ret[s] = hmm_df.loc[mask, 'log_ret'].mean()
                    state_cnt[s] = int(mask.sum())

            # Sort by vol: highest vol = PANIC, lowest vol = BULL/CHOPPY
            sorted_by_vol = sorted(state_vol.keys(), key=lambda s: state_vol[s], reverse=True)
            regime_map    = {}
            regime_map[sorted_by_vol[0]] = 'PANIC'   # highest vol
            remaining = sorted_by_vol[1:]
            if state_ret[remaining[0]] > state_ret[remaining[1]]:
                regime_map[remaining[0]] = 'BULL'
                regime_map[remaining[1]] = 'CHOPPY'
            else:
                regime_map[remaining[1]] = 'BULL'
                regime_map[remaining[0]] = 'CHOPPY'

            # Current regime = state of the last day
            current_state  = int(states[-1])
            current_regime = regime_map.get(current_state, 'CHOPPY')

            # ── Update cache ─────────────────────────────────────────────────
            self._cached_regime = current_regime
            self._cached_mult   = cfg.REGIME_MULTIPLIERS.get(current_regime, 0.5)
            self._cache_date    = datetime.utcnow().date()

            # Stats for dashboard
            self._regime_stats = {
                'regime'      : current_regime,
                'multiplier'  : self._cached_mult,
                'hmm_states'  : {
                    regime_map[s]: {
                        'avg_vol': round(state_vol[s], 4),
                        'avg_ret': round(state_ret[s], 6),
                        'count'  : state_cnt[s],
                        'pct'    : round(state_cnt[s] / len(states) * 100, 1),
                    }
                    for s in regime_map
                },
                'wti_price'   : float(df['Close'].iloc[-1]),
                'refreshed_at': datetime.utcnow().isoformat(),
            }

            logger.info(f"[DailyRegime] Current regime: {current_regime} "
                        f"(mult={self._cached_mult}) | "
                        f"WTI=${float(df['Close'].iloc[-1]):.2f}")

        except Exception as e:
            logger.error(f"[DailyRegime] Error during regime detection: {e}")
            # Keep stale cache — don't crash the paper trader

    def _fallback_regime(self, prices: pd.Series) -> str:
        """
        Simple fallback when hmmlearn is not available.
        Uses rolling realized vol to classify regime.
        """
        log_ret  = np.log(prices / prices.shift(1)).dropna()
        vol_20   = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
        if vol_20 > 0.40:
            return 'PANIC'
        elif vol_20 > 0.25:
            return 'CHOPPY'
        else:
            return 'BULL'
