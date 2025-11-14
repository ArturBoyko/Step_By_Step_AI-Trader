# bin/visualize_latent.py
"""
Візуалізація латентного простору енкодера за допомогою t-SNE.

Запуск:
    python bin/visualize_latent.py

Працює так:
    - бере checkpoints/encoder_best.pt
    - бере перший шард з out_shards/dataset_part_0000.npz
    - проганяє N прикладів через енкодер
    - робить t-SNE до 2D
    - зберігає latent_tsne.png у корені проєкту
"""

import os
import sys
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)

# Шляхи за замовчуванням
CHECKPOINT_ENCODER = os.path.join(ROOT_DIR, "checkpoints", "encoder_best.pt")
SHARD_PATH = os.path.join(ROOT_DIR, "out_shards", "dataset_part_0000.npz")
OUT_PNG = os.path.join(ROOT_DIR, "latent_tsne.png")

# Підключаємо src/
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import Encoder1DCNN_LSTM  # використовуємо тільки енкодер


def main():
    if not os.path.exists(CHECKPOINT_ENCODER):
        raise FileNotFoundError(f"Encoder checkpoint not found: {CHECKPOINT_ENCODER}")
    if not os.path.exists(SHARD_PATH):
        raise FileNotFoundError(f"Sample shard not found: {SHARD_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for visualization: {device}")

    # Завантажуємо шард
    print(f"Loading shard: {SHARD_PATH}")
    with np.load(SHARD_PATH) as z:
        X = z["X"]        # (N, 96, 34)
        yb = z["y_bin"]   # (N,)
    N_total = X.shape[0]
    N_samples = min(10_000, N_total)
    print(f"Total samples in shard: {N_total}, using first {N_samples} for t-SNE")

    X = X[:N_samples]
    yb = yb[:N_samples]

    # Завантажуємо енкодер
    print(f"Loading encoder weights from: {CHECKPOINT_ENCODER}")
    enc = Encoder1DCNN_LSTM(in_channels=34, latent_dim=128)
    state_dict = torch.load(CHECKPOINT_ENCODER, map_location=device)
    enc.load_state_dict(state_dict)
    enc.to(device)
    enc.eval()

    # Проганяємо через енкодер батчами
    batch_size = 512
    latents = []
    with torch.no_grad():
        for start in range(0, N_samples, batch_size):
            end = start + batch_size
            xb = torch.from_numpy(X[start:end]).to(device)  # (B, 96, 34)
            zb = enc(xb)                                   # (B, 128)
            latents.append(zb.cpu().numpy())
    latents = np.concatenate(latents, axis=0)  # (N_samples, 128)

    print("Latent shape:", latents.shape)
    print("Running t-SNE (this may take some time)...")

    tsne = TSNE(
        n_components=2,
        perplexity=50,
        learning_rate=200,
        verbose=1,
        random_state=42,
        init="random",
    )
    latent_2d = tsne.fit_transform(latents)

    # Малюємо scatter plot: колір = yb (0/1)
    plt.figure(figsize=(8, 8))
    # 0 = падіння, 1 = зростання
    colors = ["red" if label == 0 else "blue" for label in yb]
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, alpha=0.5, s=5)
    plt.title("t-SNE of encoder latent space (blue=up, red=down)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved t-SNE plot to: {OUT_PNG}")


if __name__ == "__main__":
    main()
