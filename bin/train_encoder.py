# bin/train_encoder.py
"""
Тренування енкодера 1D CNN -> LSTM + подвійна голова (bin + reg)
БЕЗ параметрів командного рядка.

Запуск:
    python bin/train_encoder.py

Налаштування робляться у словнику CONFIG нижче.
"""

import os
import glob
import json
import math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import sys

# === КОНФІГ (редагуємо тут при потребі) ======================================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)

CONFIG = {
    # де лежать шардовані дані
    "shards_dir": os.path.join(ROOT_DIR, "out_shards"),

    # навчання
    "epochs": 10,          # можна трохи збільшити, щоб краще збігалося
    "batch_size": 1024,
    "lr": 1e-3,
    "val_ratio": 0.05,     # частка останніх шардів для валідації (майбутнє)
    "limit_files": None,   # наприклад, 10 — щоб тренуватися тільки на перших 10 файлах

    # модель
    "latent_dim": 128,

    # ресурси
    "num_workers": 0,

    # лосс
    "alpha": 0.3,          # вага MSE частини (менше → більше уваги до BCE/класифікації)
    "scale": 10000.0,      # масштаб для y_cont (переводимо в "піпи")
    "clip_abs": 0.002,     # кліппінг сирого y_cont: ±0.002 ≈ ±20 піпів

    # куди класти чекпоїнти
    "save_dir": os.path.join(ROOT_DIR, "checkpoints"),
}
# ==============================================================================

# Підключаємо src/
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import Encoder1DCNN_LSTM, PredictionHead


class ShardedWindows(IterableDataset):
    """
    Ітератор по NPZ-шардам.
    - перемішує індекси всередині кожного шарда;
    - кліпує регресійну ціль (±clip_abs);
    - масштабує регресійну ціль (scale), щоб MSE вносив відчутний вклад у лосс.
    """
    def __init__(self, shard_paths, scale: float = 10000.0, clip_abs: float = 0.002):
        super().__init__()
        self.paths = list(shard_paths)
        self.scale = float(scale)
        self.clip_abs = float(clip_abs)

    def __iter__(self):
        for p in self.paths:
            with np.load(p) as z:
                X  = z["X"]       # (N, 96, 34)
                yb = z["y_bin"]   # (N,)
                yc = z["y_cont"]  # (N,) сире (close/open - 1)
                idx = np.arange(len(yb))
                np.random.shuffle(idx)
                for i in idx:
                    # 1) сирий таргет
                    yc_raw = float(yc[i])
                    # 2) кліппінг ±clip_abs (щоб новинні свічки не рвали MSE)
                    yc_clipped = np.clip(yc_raw, -self.clip_abs, self.clip_abs)
                    # 3) масштабування до "піпів" (scale)
                    yc_scaled = np.float32(yc_clipped * self.scale)
                    yield (
                        torch.from_numpy(X[i]),
                        torch.tensor(int(yb[i]), dtype=torch.long),
                        torch.from_numpy(np.array(yc_scaled, dtype=np.float32)),
                    )


def evaluate(enc, head, loader, device):
    enc.eval()
    head.eval()
    n = 0
    loss_sum = mae_sum = bce_sum = acc_sum = 0.0
    with torch.no_grad():
        for X, yb, yc in loader:
            X  = X.to(device)
            yb = yb.float().to(device)
            yc = yc.to(device)      # вже scaled та clipped

            z = enc(X)
            logit, pred_c = head(z)

            bce = F.binary_cross_entropy_with_logits(logit, yb)
            mse = F.mse_loss(pred_c, yc)
            loss = bce + mse        # для репорту беремо bce+mse 1:1
            mae = torch.mean(torch.abs(pred_c - yc))

            pred_bin = (torch.sigmoid(logit) >= 0.5).float()
            acc = (pred_bin == yb).float().mean()

            bs = X.size(0)
            n += bs
            loss_sum += loss.item() * bs
            mae_sum  += mae.item()  * bs
            bce_sum  += bce.item()  * bs
            acc_sum  += acc.item()  * bs

    if n == 0:
        return {
            "val_loss": float("nan"),
            "val_mae": float("nan"),
            "val_bce": float("nan"),
            "val_acc": float("nan"),
        }

    return {
        "val_loss": loss_sum / n,
        "val_mae":  mae_sum  / n,   # у тих же одиницях, що й yc (тобто ~піпи)
        "val_bce":  bce_sum  / n,
        "val_acc":  acc_sum  / n,
    }


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    all_paths = sorted(glob.glob(os.path.join(cfg["shards_dir"], "dataset_part_*.npz")))
    if cfg["limit_files"] is not None:
        all_paths = all_paths[:cfg["limit_files"]]
    if not all_paths:
        raise RuntimeError(f"No shards found in {cfg['shards_dir']}")

    # валідуємо на останніх файлах (майбутнє), тренуємось на попередніх (минуле)
    n_val = max(1, int(math.ceil(len(all_paths) * cfg["val_ratio"])))
    val_paths   = all_paths[-n_val:]
    train_paths = all_paths[:-n_val]
    if not train_paths:
        raise RuntimeError("Not enough shards for training after split. Reduce val_ratio or provide more shards.")

    print("=== TRAIN ENCODER ===")
    print(f"Device      : {device}")
    print(f"Shards total: {len(all_paths)} (train={len(train_paths)}, val={len(val_paths)})")
    print(f"Config      : {json.dumps(cfg, indent=2)}")
    print("======================")

    train_loader = DataLoader(
        ShardedWindows(train_paths, scale=cfg["scale"], clip_abs=cfg["clip_abs"]),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        ShardedWindows(val_paths, scale=cfg["scale"], clip_abs=cfg["clip_abs"]),
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    enc  = Encoder1DCNN_LSTM(in_channels=34, latent_dim=cfg["latent_dim"]).to(device)
    head = PredictionHead(latent_dim=cfg["latent_dim"]).to(device)

    opt = AdamW(list(enc.parameters()) + list(head.parameters()),
                lr=cfg["lr"], weight_decay=1e-4)

    best = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        enc.train()
        head.train()
        for step, (X, yb, yc) in enumerate(train_loader, start=1):
            X  = X.to(device)
            yb = yb.float().to(device)
            yc = yc.to(device)  # scaled & clipped

            z = enc(X)
            logit, pred_c = head(z)

            bce = F.binary_cross_entropy_with_logits(logit, yb)
            mse = F.mse_loss(pred_c, yc)
            loss = bce + cfg["alpha"] * mse   # тут вже балансимо: alpha=0.3

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            if step % 200 == 0:
                print(f"[epoch {epoch}] step {step} loss={loss.item():.6f}", flush=True)

        # Валідація
        metrics = evaluate(enc, head, val_loader, device)
        print(
            f"[epoch {epoch}] "
            f"val_loss={metrics['val_loss']:.6f} | "
            f"val_mae={metrics['val_mae']:.6f} | "
            f"val_bce={metrics['val_bce']:.6f} | "
            f"val_acc={metrics['val_acc']:.4f}",
            flush=True,
        )

        # Сейв найкращих ваг
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            torch.save(enc.state_dict(), os.path.join(cfg["save_dir"], "encoder_best.pt"))
            torch.save(head.state_dict(), os.path.join(cfg["save_dir"], "head_best.pt"))
            with open(os.path.join(cfg["save_dir"], "metrics_best.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    torch.save(enc.state_dict(), os.path.join(cfg["save_dir"], "encoder_final.pt"))
    print("Saved encoder weights:", os.path.join(cfg["save_dir"], "encoder_final.pt"))


if __name__ == "__main__":
    main()
