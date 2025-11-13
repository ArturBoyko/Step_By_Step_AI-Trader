import argparse, json, os
import numpy as np, pandas as pd
from features_engine import prepare_features
from utils import drop_warmup, write_sharded, build_windows_iter

def build_dataset(input_csv: str, output_npz: str, output_meta: str,
                  W: int = 96, stride: int = 1, include_positional: bool = True,
                  sep: str = ",", decimal: str = ".", dtype: str = "float32",
                  sharded: bool = False, out_dir: str = None, shard_size: int = 100_000):
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"Input CSV not found:\n  {input_csv}")

    raw = pd.read_csv(input_csv, sep=sep, decimal=decimal)
    raw.rename(columns={c: c.lower() for c in raw.columns}, inplace=True)
    required = {"open","high","low","close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {sorted(required)}. Missing: {sorted(missing)}")

    feat_all = prepare_features(raw)
    feat_clean = drop_warmup(feat_all)

    if sharded:
        if not out_dir:
            raise ValueError("--sharded requires --out_dir")
        meta = write_sharded(feat_clean, out_dir=out_dir, shard_size=shard_size,
                             W=W, stride=stride, include_positional=include_positional, dtype=dtype)
        with open(os.path.join(out_dir, "manifest_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {"mode": "sharded", **meta, "rows_in": int(len(raw)), "rows_after_warmup": int(len(feat_clean))}

    # non-sharded (small datasets only)
    dt = np.float32 if dtype == "float32" else np.float64
    X_list, yb_list, yc_list = [], [], []
    C = None
    for win, yb, yc, cols in build_windows_iter(feat_clean, W=W, stride=stride, include_positional=include_positional, dtype=dt):
        if C is None:
            C = win.shape[1]
        X_list.append(win); yb_list.append(yb); yc_list.append(yc)
    if not X_list:
        X = np.zeros((0, W, C or (len(cols) + (3 if include_positional else 0))), dtype=dt)
        y_bin = np.zeros((0,), dtype=np.int8)
        y_cont = np.zeros((0,), dtype=np.float32)
    else:
        X = np.stack(X_list, axis=0).astype(dt, copy=False)
        y_bin = np.array(yb_list, dtype=np.int8)
        y_cont = np.array(yc_list, dtype=np.float32)

    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)
    np.savez_compressed(output_npz, X=X, y_bin=y_bin, y_cont=y_cont)
    meta = {
        "feat_cols": cols, "W": W, "stride": stride, "include_positional": include_positional,
        "channels_per_timestep": int(X.shape[2]) if X.size else None,
        "N": int(len(y_bin)), "dtype": dtype
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"mode": "single", "X_shape": tuple(X.shape), "y_bin_mean": float(y_bin.mean()) if y_bin.size else None,
            "rows_in": int(len(raw)), "rows_after_warmup": int(len(feat_clean))}

def main():
    ap = argparse.ArgumentParser(description="Build AI trading dataset from OHLC[V] CSV")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_npz", required=False, help="Output NPZ (for non-sharded mode)")
    ap.add_argument("--output_meta", required=False, help="Output JSON (for non-sharded mode)")
    ap.add_argument("--W", type=int, default=96)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-positional", action="store_true")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--decimal", default=".")
    ap.add_argument("--dtype", default="float32", choices=["float32","float64"])
    ap.add_argument("--sharded", action="store_true", help="Write dataset in shards to --out_dir")
    ap.add_argument("--out_dir", default=None, help="Directory for shards (required if --sharded)")
    ap.add_argument("--shard_size", type=int, default=100000)
    args = ap.parse_args()

    if args.sharded:
        stats = build_dataset(args.input, None, None, W=args.W, stride=args.stride,
                              include_positional=(not args.no_positional),
                              sep=args.sep, decimal=args.decimal, dtype=args.dtype,
                              sharded=True, out_dir=args.out_dir, shard_size=args.shard_size)
    else:
        output_meta = args.output_meta or (args.output_npz.replace(".npz","_meta.json"))
        stats = build_dataset(args.input, args.output_npz, output_meta, W=args.W, stride=args.stride,
                              include_positional=(not args.no_positional),
                              sep=args.sep, decimal=args.decimal, dtype=args.dtype,
                              sharded=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
