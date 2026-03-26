import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_all_features(df_1h: pd.DataFrame,
                     df_4h: pd.DataFrame,
                     df_1d: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 31 features on live data.
    Mirrors exactly what was done in Phase 2.
    """
    df = df_1h.copy()

    # ── Returns ──
    df["return_1"]  = df["close"].pct_change(1)
    df["return_3"]  = df["close"].pct_change(3)
    df["return_6"]  = df["close"].pct_change(6)
    df["return_12"] = df["close"].pct_change(12)
    df["return_24"] = df["close"].pct_change(24)

    # ── Trend ──
    df["ema_9"]  = df["close"].ewm(span=9,  adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    df["ema_9_21_cross"]  = df["ema_9"]  - df["ema_21"]
    df["ema_21_50_cross"] = df["ema_21"] - df["ema_50"]
    df["price_vs_ema21"]  = (df["close"] - df["ema_21"]) / df["ema_21"]
    df["price_vs_ema50"]  = (df["close"] - df["ema_50"]) / df["ema_50"]

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Momentum ──
    df["rsi_7"]  = compute_rsi(df["close"], 7)
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["rsi_21"] = compute_rsi(df["close"], 21)
    df["rsi_divergence"] = df["rsi_7"] - df["rsi_21"]
    df["roc_6"]  = df["close"].pct_change(6)  * 100
    df["roc_12"] = df["close"].pct_change(12) * 100
    df["roc_24"] = df["close"].pct_change(24) * 100

    # ── Volatility ──
    df["bb_mid"]   = df["close"].rolling(20).mean()
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    bb_range       = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close  = (df["low"]  - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    # ── Volume ──
    df["vol_ma_24"] = df["volume"].rolling(24).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_24"].replace(0, np.nan)
    df["vol_ratio"] = df["vol_ratio"].fillna(1.0)

    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    obv_mean  = df["obv"].rolling(24).mean()
    obv_std   = df["obv"].rolling(24).std()
    df["obv_zscore"] = np.where(
        obv_std > 1e-8,
        (df["obv"] - obv_mean) / obv_std,
        0.0
    )
    df["obv_zscore"] = pd.Series(
        df["obv_zscore"], index=df.index
    ).fillna(0.0)

    # ── 4h features ──
    df_4h = df_4h.copy()
    if df_4h.index.tz is not None:
        df_4h.index = df_4h.index.tz_localize(None)
    df_4h.index = pd.to_datetime(df_4h.index)

    df_4h["ema_21_4h"] = df_4h["close"].ewm(span=21, adjust=False).mean()
    df_4h["ema_50_4h"] = df_4h["close"].ewm(span=50, adjust=False).mean()
    df_4h["rsi_14_4h"] = compute_rsi(df_4h["close"], 14)
    df_4h["trend_4h"]  = (df_4h["ema_21_4h"] > df_4h["ema_50_4h"]).astype(int)
    df_4h["macd_hist_4h"] = (
        df_4h["close"].ewm(span=12, adjust=False).mean() -
        df_4h["close"].ewm(span=26, adjust=False).mean()
    )

    cols_4h = ["ema_21_4h","rsi_14_4h","trend_4h","macd_hist_4h"]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)

    for col in cols_4h:
        df[col] = df_4h[col].reindex(df.index, method="ffill").bfill()

    # ── Daily regime ──
    df_1d = df_1d.copy()
    if df_1d.index.tz is not None:
        df_1d.index = df_1d.index.tz_localize(None)
    df_1d.index = pd.to_datetime(df_1d.index).normalize()

    df_1d["ema_50_1d"]  = df_1d["close"].ewm(span=50,  adjust=False).mean()
    df_1d["ema_200_1d"] = df_1d["close"].ewm(span=200, adjust=False).mean()
    df_1d["ema_200_1d"] = df_1d["ema_200_1d"].fillna(df_1d["ema_50_1d"])
    df_1d["bull_regime"] = (
        (df_1d["close"]     > df_1d["ema_50_1d"]) &
        (df_1d["ema_50_1d"] > df_1d["ema_200_1d"])
    ).astype(int)
    df_1d["rsi_14_1d"]     = compute_rsi(df_1d["close"], 14)
    df_1d["daily_atr_pct"] = (df_1d["high"] - df_1d["low"]) / df_1d["close"]

    cols_1d = ["bull_regime","rsi_14_1d","daily_atr_pct"]
    df_1h_dates = df.index.normalize()
    for col in cols_1d:
        df[col] = df_1d[col].reindex(df_1h_dates).values
        df[col] = df[col].ffill().bfill()

    return df


FEATURE_COLS = [
    "return_1","return_3","return_6","return_12","return_24",
    "ema_9_21_cross","ema_21_50_cross","price_vs_ema21","price_vs_ema50",
    "macd","macd_signal","macd_hist",
    "rsi_7","rsi_14","rsi_21","rsi_divergence",
    "roc_6","roc_12","roc_24",
    "bb_width","bb_position","atr_pct",
    "vol_ratio","obv_zscore",
    "ema_21_4h","rsi_14_4h","trend_4h","macd_hist_4h",
    "bull_regime","rsi_14_1d","daily_atr_pct",
]