import numpy as np
import pandas as pd

FEATURE_LIST = [
    "ret_intra","logret_close","range_rel","body_rel","upper_wick_rel","lower_wick_rel","vol_log","vol_chg",
    "open_rel_ema21","high_rel_ema21","low_rel_ema21","close_rel_ema21","close_rel_bbm","atr_rel",
    "sig_ema_cross","sig_rsi_zones","sig_stoch_zone_cross","sig_bb_extreme","sig_psar_flip","sig_cci_zero","sig_ha_ema",
    "sig_ema_cross_decay3","sig_rsi_zones_decay3","sig_stoch_zone_cross_decay3","sig_bb_extreme_decay3","sig_psar_flip_decay3","sig_cci_zero_decay3","sig_ha_ema_decay3",
    "flt_atr_ok","flt_adx_trend","flt_vol_spike"
]

def drop_warmup(df: pd.DataFrame) -> pd.DataFrame:
    needed = FEATURE_LIST + ["open","close"]
    return df.dropna(subset=needed).reset_index(drop=True)

def positional_channels(window_len: int) -> np.ndarray:
    pos = np.arange(window_len, dtype=np.float32) / (window_len - 1 if window_len > 1 else 1.0)
    pos_lin = pos
    pos_sin = np.sin(2 * np.pi * pos)
    pos_cos = np.cos(2 * np.pi * pos)
    return np.stack([pos_lin, pos_sin, pos_cos], axis=1).astype(np.float32)  # (W,3)

def build_windows_iter(df_feat: pd.DataFrame, W=96, stride=1, include_positional=True, dtype=np.float32):
    """Генератор, що повертає по одному вікну та таргети (поточно для t+1)."""
    close = df_feat["close"].to_numpy()
    open_ = df_feat["open"].to_numpy()
    feat_cols = [c for c in df_feat.columns if c in FEATURE_LIST]
    F = df_feat[feat_cols].to_numpy(dtype=dtype, copy=False)
    T, C = F.shape
    max_start = (T - 1) - W  # мусить існувати наступний бар
    if max_start < 0:
        return
    pos = positional_channels(W) if include_positional else None
    for s in range(0, max_start + 1, stride):
        e = s + W
        win = F[s:e, :]
        if include_positional:
            win = np.concatenate([win, pos], axis=1).astype(dtype, copy=False)
        ratio = close[e] / (open_[e] + 1e-12)
        yb = np.int8(ratio > 1.0)
        yc = np.float32(ratio - 1.0)
        yield win, yb, yc, feat_cols

def write_sharded(df_feat: pd.DataFrame, out_dir: str, shard_size: int = 100_000,
                  W: int = 96, stride: int = 1, include_positional: bool = True,
                  dtype: str = "float32"):
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    dt = np.float32 if dtype == "float32" else np.float64

    it = build_windows_iter(df_feat, W=W, stride=stride, include_positional=include_positional, dtype=dt)

    shard_idx = 0
    X_buf, yb_buf, yc_buf = [], [], []
    feat_cols = None
    total = 0

    for win, yb, yc, cols in it:
        feat_cols = cols
        X_buf.append(win)
        yb_buf.append(yb)
        yc_buf.append(yc)
        if len(X_buf) >= shard_size:
            X = np.stack(X_buf, axis=0).astype(dt, copy=False)
            y_bin = np.array(yb_buf, dtype=np.int8)
            y_cont = np.array(yc_buf, dtype=np.float32)
            np.savez_compressed(os.path.join(out_dir, f"dataset_part_{shard_idx:04d}.npz"),
                                X=X, y_bin=y_bin, y_cont=y_cont)
            total += len(X_buf)
            X_buf, yb_buf, yc_buf = [], [], []
            shard_idx += 1

    # last shard
    if X_buf:
        X = np.stack(X_buf, axis=0).astype(dt, copy=False)
        y_bin = np.array(yb_buf, dtype=np.int8)
        y_cont = np.array(yc_buf, dtype=np.float32)
        np.savez_compressed(os.path.join(out_dir, f"dataset_part_{shard_idx:04d}.npz"),
                            X=X, y_bin=y_bin, y_cont=y_cont)
        total += len(X_buf)

    meta = {
        "W": W, "stride": stride, "include_positional": include_positional,
        "dtype": dtype, "total_windows": int(total),
        "feature_cols": feat_cols, "shard_size": int(shard_size)
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta
