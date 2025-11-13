# bin/train_encoder.py
import os, glob, json, argparse, math
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import sys

CUR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(CUR), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model import Encoder1DCNN_LSTM, PredictionHead

class ShardedWindows(IterableDataset):
    def __init__(self, out_dir, paths):
        self.paths = paths
    def __iter__(self):
        for p in self.paths:
            with np.load(p) as z:
                X = z["X"]; yb = z["y_bin"]; yc = z["y_cont"]
                for i in range(len(yb)):
                    yield torch.from_numpy(X[i]), torch.tensor(int(yb[i]), dtype=torch.long), torch.from_numpy(np.array(yc[i], dtype=np.float32))

def evaluate(enc, head, loader, device):
    enc.eval(); head.eval()
    n=0; loss_sum=0.0; mae_sum=0.0; bce_sum=0.0
    with torch.no_grad():
        for X, yb, yc in loader:
            X = X.to(device)
            yb = yb.float().to(device)
            yc = yc.to(device)
            z = enc(X)
            logit, pred_c = head(z)
            bce = F.binary_cross_entropy_with_logits(logit, yb)
            mse = F.mse_loss(pred_c, yc)
            loss = bce + mse
            mae = torch.mean(torch.abs(pred_c - yc))
            bs = X.size(0)
            n += bs; loss_sum += loss.item()*bs; mae_sum += mae.item()*bs; bce_sum += bce.item()*bs
    return {"val_loss": loss_sum/max(1,n), "val_mae": mae_sum/max(1,n), "val_bce": bce_sum/max(1,n)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument("--save_dir", default="./checkpoints")
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    all_paths = sorted(glob.glob(os.path.join(args.shards_dir, "dataset_part_*.npz")))
    if args.limit_files is not None:
        all_paths = all_paths[:args.limit_files]
    if not all_paths:
        raise RuntimeError(f"No shards found in {args.shards_dir}")

    n_val = max(1, int(math.ceil(len(all_paths)*args.val_ratio)))
    val_paths = all_paths[:n_val]
    train_paths = all_paths[n_val:]

    train_loader = DataLoader(ShardedWindows(args.shards_dir, train_paths), batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader   = DataLoader(ShardedWindows(args.shards_dir, val_paths),   batch_size=args.batch_size, num_workers=args.num_workers)

    enc = Encoder1DCNN_LSTM(in_channels=34, latent_dim=args.latent_dim).to(device)
    head = PredictionHead(latent_dim=args.latent_dim).to(device)
    opt = AdamW(list(enc.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=1e-4)

    best = float("inf")
    for epoch in range(1, args.epochs+1):
        enc.train(); head.train()
        for step, (X, yb, yc) in enumerate(train_loader, start=1):
            X = X.to(device); yb = yb.float().to(device); yc = yc.to(device)
            z = enc(X)
            logit, pred_c = head(z)
            bce = F.binary_cross_entropy_with_logits(logit, yb)
            mse = F.mse_loss(pred_c, yc)
            loss = bce + mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            if step % 200 == 0:
                print(f"[epoch {epoch}] step {step} loss={loss.item():.6f}", flush=True)

        metrics = evaluate(enc, head, val_loader, device)
        print(f"[epoch {epoch}] val_loss={metrics['val_loss']:.6f} | val_mae={metrics['val_mae']:.6f} | val_bce={metrics['val_bce']:.6f}", flush=True)
        if metrics["val_loss"] < best:
            best = metrics["val_loss"]
            torch.save(enc.state_dict(), os.path.join(args.save_dir, "encoder_best.pt"))
            torch.save(head.state_dict(), os.path.join(args.save_dir, "head_best.pt"))
            with open(os.path.join(args.save_dir, "metrics_best.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    torch.save(enc.state_dict(), os.path.join(args.save_dir, "encoder_final.pt"))
    print("Saved encoder weights:", os.path.join(args.save_dir, "encoder_final.pt"))

if __name__ == "__main__":
    main()
