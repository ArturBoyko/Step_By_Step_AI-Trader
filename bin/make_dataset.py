# bin/make_dataset.py
"""
Автоматичний билд датасету без параметрів командного рядка.

Запуск:
    python bin/make_dataset.py

Налаштування (редагуємо тут, якщо потрібно інші шляхи/параметри):
    - INPUT_CSV: шлях до сирих котирувань M1 EURUSD
    - OUT_DIR: папка для шардованого датасету
    - W, STRIDE, SHARD_SIZE, DTYPE: параметри побудови
"""

import os
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)  # корінь проєкту (де src, bin, data, out_shards)

# ШЛЯХИ (можна змінити під себе)
INPUT_CSV = os.path.join(ROOT_DIR, "data", "eurusd_m1_ready.csv")
OUT_DIR   = os.path.join(ROOT_DIR, "out_shards")

# ПАРАМЕТРИ ВІКНА
W = 96
STRIDE = 1
DTYPE = "float32"     # float32 достатньо
SHARD_SIZE = 100_000  # скільки вікон у одному файлі *.npz

# Підключаємо src/
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset_builder import build_dataset  # імпортуємо з вашого src/dataset_builder.py


def main():
    print("=== BUILD DATASET (SHARDED) ===")
    print(f"ROOT_DIR   : {ROOT_DIR}")
    print(f"INPUT_CSV  : {INPUT_CSV}")
    print(f"OUT_DIR    : {OUT_DIR}")
    print(f"W={W}, STRIDE={STRIDE}, DTYPE={DTYPE}, SHARD_SIZE={SHARD_SIZE}")
    print("================================")

    stats = build_dataset(
        input_csv=INPUT_CSV,
        output_npz=None,            # не потрібно в шард-режимі
        output_meta=None,           # не потрібно в шард-режимі
        W=W,
        stride=STRIDE,
        include_positional=True,
        sep=",",
        decimal=".",
        dtype=DTYPE,
        sharded=True,
        out_dir=OUT_DIR,
        shard_size=SHARD_SIZE,
    )

    print("\nГотово. Статистика:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
