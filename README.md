# AI Trading Dataset Builder (EURUSD M1)

This project builds a battle-ready training dataset from raw OHLC[V] CSV:
- 31 engineered features without lookahead
- Sliding windows (default W=96, stride=1)
- Positional channels per timestep (pos_lin, pos_sin, pos_cos)
- Targets:
  - y_bin = 1 if Close(t+1)/Open(t+1) > 1 else 0
  - y_cont = Close(t+1)/Open(t+1) - 1

## Quick start
```bash
python bin/make_dataset.py --input /path/to/eurusd_m1_ready.csv --output_npz /path/out/eurusd_W96_pos1_yNext.npz
```
