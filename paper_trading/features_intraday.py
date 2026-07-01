# ============================================================================
# PETROQUANT PAPER TRADING — INTRADAY FEATURE ENGINEERING
# ============================================================================
# Transforms raw 1-minute OHLCV bars into ML features for the XGBoost model.
#
# Features are all backward-looking (no future leakage):
#   Momentum   — returns over 1/5/15/30 minutes
#   Trend      — EMA crosses, VWAP deviation
#   Volatility — ATR, realized vol, Bollinger Band position
#   Volume     — volume ratio (spike detection)
#   Candle     — body-to-range ratio (candle shape)
#   Regime     — daily HMM state passed in as external signal
#
# build_features(df) → returns feat_df with all computed columns
# build_target(feat_df, horizon) → adds 'Target' (1=up, 0=down in N bars)
# ============================================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   regime: str = 'BULL') -> pd.DataFrame:
    """
    Build intraday ML features from 1-min OHLCV bars.

    Parameters
    ----------
    df     : pd.DataFrame  — 1-min OHLCV with columns [Open, High, Low, Close, Volume]
    regime : str           — current daily regime from HMM ('BULL'/'CHOPPY'/'PANIC')

    Returns
    -------
    pd.DataFrame — original OHLCV + all engineered features (NaN rows dropped)
    """
    if df.empty or len(df) < 30:
        logger.warning("[Features] Not enough bars to compute features (need ≥ 30)")
        return pd.DataFrame()

    feat = df.copy()
    c = feat['Close']
    o = feat['Open']
    h = feat['High']
    l = feat['Low']
    v = feat.get('Volume', pd.Series(np.ones(len(feat)), index=feat.index))

    # ── 1. Price Returns (Momentum) ──────────────────────────────────────────
    feat['Ret_1m']  = np.log(c / c.shift(1))
    feat['Ret_5m']  = np.log(c / c.shift(5))
    feat['Ret_15m'] = np.log(c / c.shift(15))
    feat['Ret_30m'] = np.log(c / c.shift(30))

    # Cumulative return direction
    feat['Ret_sign_1m']  = np.sign(feat['Ret_1m'])
    feat['Ret_sign_5m']  = np.sign(feat['Ret_5m'])

    # ── 2. Trend — EMA ──────────────────────────────────────────────────────
    feat['EMA_5']    = c.ewm(span=5,  adjust=False).mean()
    feat['EMA_20']   = c.ewm(span=20, adjust=False).mean()
    feat['EMA_50']   = c.ewm(span=50, adjust=False).mean()

    # EMA cross signals
    feat['EMA_5_20_cross']  = (feat['EMA_5']  - feat['EMA_20']) / c   # normalized
    feat['EMA_5_50_cross']  = (feat['EMA_5']  - feat['EMA_50']) / c
    feat['EMA_20_50_cross'] = (feat['EMA_20'] - feat['EMA_50']) / c

    # Price vs EMAs
    feat['Price_vs_EMA5']  = (c - feat['EMA_5'])  / c
    feat['Price_vs_EMA20'] = (c - feat['EMA_20']) / c

    # ── 3. RSI (14-period, on 1-min bars) ────────────────────────────────────
    feat['RSI_14'] = _rsi(c, period=14)
    feat['RSI_7']  = _rsi(c, period=7)

    # RSI derivative (momentum of momentum)
    feat['RSI_14_chg'] = feat['RSI_14'].diff(3)

    # ── 4. ATR — Volatility ───────────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    feat['ATR_14'] = tr.ewm(span=14, adjust=False).mean()

    # Normalize ATR as fraction of price
    feat['ATR_14_pct'] = feat['ATR_14'] / c

    # ── 5. VWAP Deviation ────────────────────────────────────────────────────
    # VWAP resets at each session start — we approximate with rolling 60-bar VWAP
    vwap_window = 60
    tp = (h + l + c) / 3  # typical price
    feat['VWAP_60'] = (tp * v).rolling(vwap_window).sum() / v.rolling(vwap_window).sum()
    feat['VWAP_dev'] = (c - feat['VWAP_60']) / feat['ATR_14'].replace(0, np.nan)
    feat['VWAP_dev'] = feat['VWAP_dev'].clip(-5, 5)  # winsorize extreme deviations

    # ── 6. Bollinger Bands ───────────────────────────────────────────────────
    bb_window = 20
    feat['BB_mid']   = c.rolling(bb_window).mean()
    feat['BB_std']   = c.rolling(bb_window).std()
    feat['BB_upper'] = feat['BB_mid'] + 2 * feat['BB_std']
    feat['BB_lower'] = feat['BB_mid'] - 2 * feat['BB_std']
    bb_range = (feat['BB_upper'] - feat['BB_lower']).replace(0, np.nan)
    feat['BB_pos']   = (c - feat['BB_lower']) / bb_range   # 0=bottom, 1=top
    feat['BB_width'] = bb_range / c                          # bandwidth as % of price

    # ── 7. Volume Features ───────────────────────────────────────────────────
    v_ma = v.rolling(20).mean().replace(0, np.nan)
    feat['Vol_ratio']  = (v / v_ma).clip(0, 10)   # volume spike: >1 = above average
    feat['Vol_sign']   = feat['Vol_ratio'] * feat['Ret_sign_1m']   # direction × volume

    # ── 8. Candle Structure ──────────────────────────────────────────────────
    candle_range = (h - l).replace(0, np.nan)
    feat['Body_ratio']   = (c - o).abs() / candle_range     # body / full range
    feat['Wick_upper']   = (h - c.clip(upper=o)) / candle_range   # upper wick fraction
    feat['Wick_lower']   = (c.clip(upper=o) - l) / candle_range   # lower wick fraction
    feat['Candle_dir']   = np.sign(c - o)                          # +1 green, -1 red

    # ── 9. Realized Volatility ───────────────────────────────────────────────
    feat['RealVol_15m'] = feat['Ret_1m'].rolling(15).std() * np.sqrt(252 * 390)  # annualized
    feat['RealVol_30m'] = feat['Ret_1m'].rolling(30).std() * np.sqrt(252 * 390)
    feat['Vol_regime']  = (feat['RealVol_15m'] / feat['RealVol_30m']).clip(0, 5)  # vol acceleration

    # ── 10. Consecutive candle count ─────────────────────────────────────────
    # How many candles in a row have been the same color (streak)
    feat['Streak'] = _streak(feat['Candle_dir'])

    # ── 11. Daily Regime (Macro filter from HMM) ─────────────────────────────
    feat['Regime_BULL']   = 1.0 if regime == 'BULL'   else 0.0
    feat['Regime_CHOPPY'] = 1.0 if regime == 'CHOPPY' else 0.0
    feat['Regime_PANIC']  = 1.0 if regime == 'PANIC'  else 0.0

    # ── Drop helper columns we don't want as model features ──────────────────
    _drop_cols = ['EMA_5', 'EMA_20', 'EMA_50', 'BB_mid', 'BB_std',
                  'BB_upper', 'BB_lower', 'VWAP_60']
    feat = feat.drop(columns=[col for col in _drop_cols if col in feat.columns])

    # ── Drop NaN rows ────────────────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"[Features] Built {len(feat)} rows × {feat.shape[1]} cols "
                f"(including {feat.shape[1]-5} engineered features)")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
def build_target(feat_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add binary classification target: will price be higher in N bars?

    Target = 1 if Close[now + horizon] > Close[now] else 0

    Parameters
    ----------
    feat_df : pd.DataFrame — feature DataFrame (output of build_features)
    horizon : int          — number of 1-min bars to look forward (default: 5)

    Returns
    -------
    pd.DataFrame — same df with added 'Target' and 'Fwd_Return' columns
                   (rows where target is NaN are dropped)
    """
    df = feat_df.copy()
    fwd_return = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Fwd_Return'] = fwd_return
    df['Target']     = (fwd_return > 0).astype(int)
    df = df.dropna(subset=['Target'])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE COLUMN NAMES (used by model to select input features)
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns list of feature column names (excludes raw OHLCV and target columns).
    """
    exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Fwd_Return'}
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (EMA-based)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _streak(direction: pd.Series) -> pd.Series:
    """
    Compute consecutive candle streak (vectorized via numpy).
    +3 means 3 green candles in a row, -2 means 2 red candles in a row.

    Uses raw numpy arrays instead of repeated pandas iloc calls to avoid
    per-row overhead that was very slow on 5000+ bar DataFrames.
    """
    arr    = direction.values.astype(float)
    n      = len(arr)
    streak = np.zeros(n, dtype=float)
    if n == 0:
        return pd.Series(streak, index=direction.index)
    streak[0] = arr[0]
    for i in range(1, n):
        if arr[i] == arr[i - 1] and arr[i] != 0:
            streak[i] = streak[i - 1] + arr[i]
        else:
            streak[i] = arr[i]
    return pd.Series(streak, index=direction.index)
