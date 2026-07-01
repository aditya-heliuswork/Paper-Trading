# ============================================================================
# PETROQUANT PAPER TRADING — INTRADAY SIGNAL MODEL
# ============================================================================
# Rolling-window XGBoost classifier for WTI price direction prediction.
# Supports all timeframes: 1m, 5m, 15m, 1h, 1d
#
# Strategy:
#   - Train on last N bars (rolling window, no walk-forward)
#   - Predict: will Close[now+horizon] > Close[now]? (1=UP, 0=DOWN)
#   - Retrain every RETRAIN_EVERY_MINS for market adaptation
#   - Returns (signal, probability) for the latest bar
#
# IntradaySignalModel:
#   train(feat_df)          — fits XGBoost on provided labeled data
#   predict_latest(feat_df) — returns (signal, prob) for last row
#   should_retrain()        — True if overdue for retraining
#   get_feature_importance()— returns pd.Series
#   get_model_status()      — returns dict with training metadata
# ============================================================================

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from . import config as cfg
from .features_intraday import get_feature_columns, build_target

logger = logging.getLogger(__name__)


class IntradaySignalModel:
    """
    Rolling-window XGBoost classifier for WTI OHLCV data.

    Predicts price direction N bars forward.
    Retrains on the latest N bars every RETRAIN_EVERY_MINS.

    Bug #8 fix: min_train_bars is now sourced from config (which is timeframe-aware)
    so the model won't refuse to train on coarser timeframes where fewer bars exist.
    """

    def __init__(self,
                 horizon: int = None,
                 min_train_bars: int = None,
                 retrain_every_mins: int = None,
                 buy_threshold: float = cfg.BUY_THRESHOLD,
                 sell_threshold: float = cfg.SELL_THRESHOLD):

        # Use values from config (which reflect the active timeframe)
        self.horizon            = horizon            or cfg.PREDICT_HORIZON
        self.min_train_bars     = min_train_bars     or cfg.MIN_TRAIN_BARS
        self.retrain_every_mins = retrain_every_mins or cfg.RETRAIN_EVERY_MINS
        self.buy_threshold      = buy_threshold
        self.sell_threshold     = sell_threshold

        # Internal state
        self._model              = None
        self._scaler             = None
        self._feature_cols       = []
        self._last_train_time    = None
        self._last_train_acc     = None
        self._n_train_bars       = 0
        self._is_trained         = False
        self._feature_importance = None

    def refresh_from_config(self) -> None:
        """
        Re-read horizon / min_train_bars / retrain_every_mins from the current
        config. Call this after apply_timeframe() to keep the model in sync.
        """
        self.horizon            = cfg.PREDICT_HORIZON
        self.min_train_bars     = cfg.MIN_TRAIN_BARS
        self.retrain_every_mins = cfg.RETRAIN_EVERY_MINS
        logger.info(f"[Model] Refreshed from config — "
                    f"horizon={self.horizon}, min_bars={self.min_train_bars}, "
                    f"retrain_mins={self.retrain_every_mins}")

    # ── Training ─────────────────────────────────────────────────────────────
    def train(self, feat_df: pd.DataFrame) -> dict:
        """
        Train XGBoost on the provided feature DataFrame.

        Parameters
        ----------
        feat_df : pd.DataFrame — output of build_features() (with or without 'Target')

        Returns
        -------
        dict — training metadata: accuracy, n_bars, feature_count
        """
        if 'Target' not in feat_df.columns:
            feat_df = build_target(feat_df, horizon=self.horizon)

        if len(feat_df) < self.min_train_bars:
            logger.warning(f"[Model] Only {len(feat_df)} bars — "
                           f"need {self.min_train_bars} to train. Skipping.")
            return {'status': 'skipped', 'reason': 'not enough bars',
                    'have': len(feat_df), 'need': self.min_train_bars}

        # ── Select features ──────────────────────────────────────────────────
        self._feature_cols = get_feature_columns(feat_df)
        X = feat_df[self._feature_cols].values
        y = feat_df['Target'].values

        # ── Train / validation split (80/20, time-ordered) ──────────────────
        split     = int(len(X) * 0.80)
        X_train   = X[:split]
        y_train   = y[:split]
        X_val     = X[split:]
        y_val     = y[split:]

        # ── Scale ────────────────────────────────────────────────────────────
        self._scaler  = StandardScaler()
        X_train_s = self._scaler.fit_transform(X_train)
        X_val_s   = self._scaler.transform(X_val)

        # ── XGBoost — calibrated for intraday noise ───────────────────────
        self._model = XGBClassifier(
            max_depth        = 3,
            n_estimators     = 200,
            learning_rate    = 0.05,
            subsample        = 0.7,
            colsample_bytree = 0.7,
            min_child_weight = 10,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            objective        = 'binary:logistic',
            eval_metric      = 'logloss',
            random_state     = 42,
            verbosity        = 0,
        )

        self._model.fit(
            X_train_s, y_train,
            eval_set = [(X_val_s, y_val)],
            verbose  = False,
        )

        # ── Validation accuracy ───────────────────────────────────────────
        val_preds = (self._model.predict_proba(X_val_s)[:, 1] > 0.5).astype(int)
        val_acc   = accuracy_score(y_val, val_preds) if len(y_val) > 0 else 0.5

        # ── Feature importance ────────────────────────────────────────────
        self._feature_importance = pd.Series(
            self._model.feature_importances_,
            index=self._feature_cols
        ).sort_values(ascending=False)

        # ── Update state ──────────────────────────────────────────────────
        self._last_train_time = datetime.utcnow()
        self._last_train_acc  = val_acc
        self._n_train_bars    = len(X_train)
        self._is_trained      = True

        acc_grade = ('GOOD' if val_acc > 0.55 else
                     'FAIR' if val_acc > 0.52 else 'WEAK')

        logger.info(f"[Model] Trained | bars={len(feat_df)} | "
                    f"val_acc={val_acc:.2%} ({acc_grade}) | "
                    f"features={len(self._feature_cols)} | "
                    f"horizon={self.horizon}")

        return {
            'status'       : 'trained',
            'val_accuracy' : round(val_acc, 4),
            'n_train_bars' : len(X_train),
            'n_val_bars'   : len(X_val),
            'n_features'   : len(self._feature_cols),
            'trained_at'   : self._last_train_time.isoformat(),
            'acc_grade'    : acc_grade,
            'top_features' : self._feature_importance.head(5).to_dict(),
        }

    # ── Prediction ───────────────────────────────────────────────────────────
    def predict_latest(self, feat_df: pd.DataFrame) -> tuple[str, float]:
        """
        Predict signal for the most recent bar.

        Parameters
        ----------
        feat_df : pd.DataFrame — feature DataFrame (from build_features)

        Returns
        -------
        tuple: (signal, probability)
            signal      : 'BUY' | 'SELL' | 'HOLD'
            probability : float 0-1 (probability of price going UP)
        """
        if not self._is_trained or self._model is None:
            logger.warning("[Model] Model not trained yet — returning HOLD")
            return ('HOLD', 0.5)

        if feat_df.empty:
            return ('HOLD', 0.5)

        # Get last row (most recent bar)
        missing_cols = [c for c in self._feature_cols if c not in feat_df.columns]
        if missing_cols:
            logger.warning(f"[Model] Missing features: {missing_cols}")
            return ('HOLD', 0.5)

        X_latest = feat_df[self._feature_cols].iloc[[-1]].values

        try:
            X_scaled = self._scaler.transform(X_latest)
            prob_up  = float(self._model.predict_proba(X_scaled)[0, 1])
        except Exception as e:
            logger.error(f"[Model] Prediction error: {e}")
            return ('HOLD', 0.5)

        # Translate probability to signal
        if prob_up > self.buy_threshold:
            signal = 'BUY'
        elif prob_up < self.sell_threshold:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return (signal, round(prob_up, 4))

    # ── Retraining check ─────────────────────────────────────────────────────
    def should_retrain(self) -> bool:
        """
        Returns True if the model should be retrained.
        Triggers if:
          - Never been trained
          - More than retrain_every_mins have passed since last training
        """
        if not self._is_trained or self._last_train_time is None:
            return True

        elapsed_mins = (datetime.utcnow() - self._last_train_time).total_seconds() / 60
        return elapsed_mins >= self.retrain_every_mins

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_feature_importance(self) -> pd.Series | None:
        """Returns feature importance Series, or None if not trained."""
        return self._feature_importance

    def get_model_status(self) -> dict:
        """Returns current model state summary."""
        if not self._is_trained:
            return {'trained': False, 'timeframe': cfg.ACTIVE_TIMEFRAME}

        elapsed = (datetime.utcnow() - self._last_train_time).total_seconds() / 60
        retrain_in = max(0, self.retrain_every_mins - elapsed)

        return {
            'trained'            : True,
            'last_train_utc'     : self._last_train_time.isoformat(),
            'val_accuracy'       : self._last_train_acc,
            'n_train_bars'       : self._n_train_bars,
            'n_features'         : len(self._feature_cols),
            'retrain_due_in_mins': round(retrain_in, 1),
            'needs_retrain'      : self.should_retrain(),
            'horizon'            : self.horizon,
            'timeframe'          : cfg.ACTIVE_TIMEFRAME,
        }
