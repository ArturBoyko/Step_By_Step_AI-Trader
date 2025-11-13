# src/features_engine.py
import pandas as pd
import numpy as np

EPS = 1e-12

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma_tp = sma(tp, period)
    mad = (tp - sma_tp).abs().rolling(window=period, min_periods=period).mean()
    cci_val = (tp - sma_tp) / (0.015 * mad)
    return cci_val

def dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(high, low, close)
    atr_wilder = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_wilder
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1/period, adjust=False).mean() / atr_wilder
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return plus_di, minus_di, adx

def heiken_ashi(open_, high, low, close):
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = pd.Series(index=open_.index, dtype=float)
    ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2.0
    for i in range(1, len(open_)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    return ha_open, ha_high, ha_low, ha_close

def psar(high: pd.Series, low: pd.Series, step: float = 0.02, max_step: float = 0.2) -> pd.Series:
    length = len(high)
    psar = pd.Series(index=high.index, dtype=float)
    bull = True
    af = step
    ep = low.iloc[0]
    psar.iloc[0] = low.iloc[0]
    for i in range(1, length):
        prev_psar = psar.iloc[i-1]
        high_i, low_i = high.iloc[i], low.iloc[i]
        high_prev, low_prev = high.iloc[i-1], low.iloc[i-1]
        psar_i = prev_psar + af * (ep - prev_psar)
        if bull:
            psar_i = min(psar_i, low_prev)
            if i >= 2:
                psar_i = min(psar_i, low.iloc[i-2])
        else:
            psar_i = max(psar_i, high_prev)
            if i >= 2:
                psar_i = max(psar_i, high.iloc[i-2])
        if bull:
            if low_i < psar_i:
                bull = False
                psar_i = ep
                ep = high_i
                af = step
            else:
                if high_i > ep:
                    ep = high_i
                    af = min(af + step, max_step)
        else:
            if high_i > psar_i:
                bull = True
                psar_i = ep
                ep = low_i
                af = step
            else:
                if low_i < ep:
                    ep = low_i
                    af = min(af + step, max_step)
        psar.iloc[i] = psar_i
    return psar

def rolling_decay(sig: pd.Series, w0=1.0, w1=0.6, w2=0.3) -> pd.Series:
    return sig * w0 + sig.shift(1).fillna(0)*w1 + sig.shift(2).fillna(0)*w2

FEATURE_LIST = [
    "ret_intra","logret_close","range_rel","body_rel","upper_wick_rel","lower_wick_rel","vol_log","vol_chg",
    "open_rel_ema21","high_rel_ema21","low_rel_ema21","close_rel_ema21","close_rel_bbm","atr_rel",
    "sig_ema_cross","sig_rsi_zones","sig_stoch_zone_cross","sig_bb_extreme","sig_psar_flip","sig_cci_zero","sig_ha_ema",
    "sig_ema_cross_decay3","sig_rsi_zones_decay3","sig_stoch_zone_cross_decay3","sig_bb_extreme_decay3","sig_psar_flip_decay3","sig_cci_zero_decay3","sig_ha_ema_decay3",
    "flt_atr_ok","flt_adx_trend","flt_vol_spike"
]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    for col in ["open","high","low","close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "volume" not in df.columns:
        df["volume"] = 1.0

    # 1) Candle geometry
    df["ret_intra"] = df["close"]/(df["open"]+EPS)-1.0
    df["logret_close"] = np.log(df["close"]/(df["close"].shift(1)+EPS))
    df["range_rel"] = (df["high"]-df["low"])/(df["close"]+EPS)
    df["body_rel"] = (df["close"]-df["open"])/(df["close"]+EPS)
    df["upper_wick_rel"] = (df["high"]-np.maximum(df["open"],df["close"]))/(df["close"]+EPS)
    df["lower_wick_rel"] = (np.minimum(df["open"],df["close"])-df["low"])/(df["close"]+EPS)
    df["vol_log"] = np.log1p(df["volume"])
    df["vol_chg"] = df["vol_log"].diff()

    # 2) Relatives
    ema21 = ema(df["close"],21)
    df["open_rel_ema21"]  = df["open"]/(ema21+EPS)-1.0
    df["high_rel_ema21"]  = df["high"]/(ema21+EPS)-1.0
    df["low_rel_ema21"]   = df["low"]/(ema21+EPS)-1.0
    df["close_rel_ema21"] = df["close"]/(ema21+EPS)-1.0

    bb_lower, bb_mid, bb_upper = bollinger_bands(df["close"],20,2.0)
    df["close_rel_bbm"] = df["close"]/(bb_mid+EPS)-1.0

    atr14 = atr(df["high"], df["low"], df["close"], 14)
    atr14_ma100 = atr14.rolling(window=100, min_periods=100).mean()
    df["atr_rel"] = atr14/(atr14_ma100+EPS)-1.0

    # 3) Signals
    ema9 = ema(df["close"],9)
    diff_ema = ema9-ema21
    prev_diff = diff_ema.shift(1)
    df["sig_ema_cross"] = np.where((diff_ema>0)&(prev_diff<=0),1, np.where((diff_ema<0)&(prev_diff>=0),-1,0))

    rsi14 = rsi(df["close"],14)
    rsi_prev = rsi14.shift(1)
    df["sig_rsi_zones"] = np.where((rsi_prev<30)&(rsi14>=30),1, np.where((rsi_prev>70)&(rsi14<=70),-1,0))

    st_k, st_d = stoch_kd(df["high"], df["low"], df["close"], 14, 3)
    st_k_prev = st_k.shift(1)
    st_d_prev = st_d.shift(1)
    cross_up  = (st_k_prev <= st_d_prev) & (st_k > st_d)
    cross_dn  = (st_k_prev >= st_d_prev) & (st_k < st_d)
    in_low_zone  = (st_k < 20) | (st_k_prev < 20)
    in_high_zone = (st_k > 80) | (st_k_prev > 80)
    df["sig_stoch_zone_cross"] = np.where(cross_up & in_low_zone, 1, np.where(cross_dn & in_high_zone, -1, 0))

    touch_lower = df["close"] <= (bb_lower + EPS)
    touch_upper = df["close"] >= (bb_upper - EPS)
    df["sig_bb_extreme"] = np.where(touch_lower & (rsi14 <= 35), 1, np.where(touch_upper & (rsi14 >= 65), -1, 0))

    psar_series = psar(df["high"], df["low"], step=0.02, max_step=0.2)
    trend_now = df["close"] > psar_series
    trend_prev = trend_now.shift(1)
    trend_prev = trend_prev.astype("boolean").fillna(bool(trend_now.iloc[0])).astype(bool)
    df["sig_psar_flip"] = np.where(trend_now & (~trend_prev), 1,
                                   np.where((~trend_now) & trend_prev, -1, 0))
    cci20 = cci(df["high"], df["low"], df["close"], 20)
    cci_prev = cci20.shift(1)
    df["sig_cci_zero"] = np.where((cci_prev <= 0) & (cci20 > 0), 1, np.where((cci_prev >= 0) & (cci20 < 0), -1, 0))

    ha_open, ha_high, ha_low, ha_close = heiken_ashi(df["open"], df["high"], df["low"], df["close"])
    ha_bull = ha_close >= ha_open
    ha_bull_prev = ha_bull.shift(1)
    ema21_for_ha = ema(df["close"], 21)
    ha_bull_prev = ha_bull_prev.astype("boolean").fillna(bool(ha_bull.iloc[0])).astype(bool)
    df["sig_ha_ema"] = np.where((~ha_bull_prev) & ha_bull & (df["close"] > ema21_for_ha), 1,
                                np.where(ha_bull_prev & (~ha_bull) & (df["close"] < ema21_for_ha), -1, 0))
    for sig_name in ["sig_ema_cross","sig_rsi_zones","sig_stoch_zone_cross","sig_bb_extreme","sig_psar_flip","sig_cci_zero","sig_ha_ema"]:
        df[f"{sig_name}_decay3"] = rolling_decay(df[sig_name])

    plus_di, minus_di, adx14 = dmi_adx(df["high"], df["low"], df["close"], 14)
    atr14_ma100_strict = atr14.rolling(window=100, min_periods=100).mean()
    df["flt_atr_ok"] = (atr14 > atr14_ma100_strict).astype(int)
    df["flt_adx_trend"] = (adx14 > 25).astype(int)
    vol_med50 = df["volume"].rolling(window=50, min_periods=50).median()
    df["flt_vol_spike"] = (df["volume"] > (1.5 * vol_med50)).astype(int)

    return df
