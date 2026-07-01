# ============================================================================
# PETROQUANT PAPER TRADING — INTRADAY FEATURE ENGINEERING
# ============================================================================
# Transforms raw OHLCV bars into ML features for the XGBoost model.
# Supports all timeframes: 1m, 5m, 15m, 1h, 1d
#
# Features are all backward-looking (no future leakage):
#   Momentum   — returns over 1/5/15/30 candles
#   Trend      — EMA crosses, VWAP deviation
#   Volatility — ATR, realized vol, Bollinger Band position
#   Volume     — volume ratio (spike detection)
#   Candle     — body-to-range ratio (candle shape)
#   Regime     — daily HMM state passed in as external signal
#
# build_features(df, regime, timeframe_minutes) → feat_df with all columns
# build_target(feat_df, horizon)               → adds 'Target' (1=up, 0=down)
# ============================================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   regime: str = 'BULL',
                   timeframe_minutes: int = 1) -> pd.DataFrame:
    """
    Build ML features from OHLCV bars (any timeframe).

    Parameters
    ----------
    df                 : pd.DataFrame  — OHLCV with [Open, High, Low, Close, Volume]
    regime             : str           — daily regime from HMM ('BULL'/'CHOPPY'/'PANIC')
    timeframe_minutes  : int           — bar duration in minutes (1, 5, 15, 60, 1440)
                                         Used to scale rolling windows appropriately.

    Returns
    -------
    pd.DataFrame — original OHLCV + all engineered features (NaN rows dropped)
    """
    # Bug #14 fix: minimum bar check raised to VWAP window (60 bars), not 30
    min_bars = 60
    if df.empty or len(df) < min_bars:
        logger.warning(f"[Features] Not enough bars to compute features "
                       f"(have {len(df)}, need ≥ {min_bars})")
        return pd.DataFrame()

    feat = df.copy()
    c = feat['Close']
    o = feat['Open']
    h = feat['High']
    l = feat['Low']
    v = feat.get('Volume', pd.Series(np.ones(len(feat)), index=feat.index))

    # ── Scale rolling windows to timeframe ───────────────────────────────────
    # For faster timeframes we keep the same window sizes (they already capture
    # meaningful market microstructure). For daily bars we scale up slightly.
    _w = timeframe_minutes  # convenience alias

    # ── 1. Price Returns (Momentum) ──────────────────────────────────────────
    feat['Ret_1']  = np.log(c / c.shift(1))
    feat['Ret_5']  = np.log(c / c.shift(5))
    feat['Ret_15'] = np.log(c / c.shift(15))
    feat['Ret_30'] = np.log(c / c.shift(30))

    # Cumulative return direction
    feat['Ret_sign_1']  = np.sign(feat['Ret_1'])
    feat['Ret_sign_5']  = np.sign(feat['Ret_5'])

    # ── 2. Trend — EMA ──────────────────────────────────────────────────────
    feat['EMA_5']    = c.ewm(span=5,  adjust=False).mean()
    feat['EMA_20']   = c.ewm(span=20, adjust=False).mean()
    feat['EMA_50']   = c.ewm(span=50, adjust=False).mean()

    # EMA cross signals (normalized by price)
    feat['EMA_5_20_cross']  = (feat['EMA_5']  - feat['EMA_20']) / c
    feat['EMA_5_50_cross']  = (feat['EMA_5']  - feat['EMA_50']) / c
    feat['EMA_20_50_cross'] = (feat['EMA_20'] - feat['EMA_50']) / c

    # Price vs EMAs
    feat['Price_vs_EMA5']  = (c - feat['EMA_5'])  / c
    feat['Price_vs_EMA20'] = (c - feat['EMA_20']) / c

    # ── 3. RSI (14-period) ────────────────────────────────────────────────────
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
    # Rolling VWAP (60-bar window) normalized by ATR
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
    feat['BB_width'] = bb_range / c                         # bandwidth as % of price

    # ── 7. Volume Features ───────────────────────────────────────────────────
    v_ma = v.rolling(20).mean().replace(0, np.nan)
    feat['Vol_ratio']  = (v / v_ma).clip(0, 10)   # volume spike: >1 = above average
    feat['Vol_sign']   = feat['Vol_ratio'] * feat['Ret_sign_1']   # direction × volume

    # ── 8. Candle Structure ──────────────────────────────────────────────────
    candle_range = (h - l).replace(0, np.nan)
    feat['Body_ratio']   = (c - o).abs() / candle_range     # body / full range
    feat['Wick_upper']   = (h - c.clip(upper=o)) / candle_range   # upper wick fraction
    feat['Wick_lower']   = (c.clip(upper=o) - l) / candle_range   # lower wick fraction
    feat['Candle_dir']   = np.sign(c - o)                         # +1 green, -1 red

    # ── 9. Realized Volatility ───────────────────────────────────────────────
    # Annualization factor: bars_per_year = 252 * trading_bars_per_day
    # 1m: 252*390=98280; 5m: 252*78=19656; 15m: 252*26=6552; 1h: 252*6.5=1638; 1d: 252
    ann_factor = {1: 252*390, 5: 252*78, 15: 252*26, 60: 252*7, 1440: 252}
    ann = ann_factor.get(_w, 252*390)

    feat['RealVol_15'] = feat['Ret_1'].rolling(15).std() * np.sqrt(ann)
    feat['RealVol_30'] = feat['Ret_1'].rolling(30).std() * np.sqrt(ann)
    feat['Vol_regime'] = (feat['RealVol_15'] / feat['RealVol_30']).clip(0, 5)

    # ── 10. Consecutive candle streak ────────────────────────────────────────
    feat['Streak'] = _streak(feat['Candle_dir'])

    # ── 11. Daily Regime (Macro filter from HMM) ─────────────────────────────
    feat['Regime_BULL']   = 1.0 if regime == 'BULL'   else 0.0
    feat['Regime_CHOPPY'] = 1.0 if regime == 'CHOPPY' else 0.0
    feat['Regime_PANIC']  = 1.0 if regime == 'PANIC'  else 0.0

    # ── Drop helper columns not needed as model features ─────────────────────
    _drop_cols = ['EMA_5', 'EMA_20', 'EMA_50', 'BB_mid', 'BB_std',
                  'BB_upper', 'BB_lower', 'VWAP_60']
    feat = feat.drop(columns=[col for col in _drop_cols if col in feat.columns])

    # ── Drop NaN / inf rows ──────────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"[Features] Built {len(feat)} rows × {feat.shape[1]} cols "
                f"(timeframe={_w}m, {feat.shape[1]-5} engineered features)")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
def build_target(feat_df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Add binary classification target: will price be higher in N bars?

    Target = 1 if Close[now + horizon] > Close[now] else 0

    Parameters
    ----------
    feat_df : pd.DataFrame — feature DataFrame (output of build_features)
    horizon : int          — number of bars to look forward (default: 5)

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
