import os
import sys
import csv
import pickle
import numpy as np
import neat
import matplotlib.pyplot as plt

# === Пути =====================================================================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)
NEAT_RUNS_DIR = os.path.join(ROOT_DIR, "neat_runs")

LATENTS_CACHE = os.path.join(NEAT_RUNS_DIR, "latents_cache.npz")

# Попробуем несколько вариантов имени winner-файла
WINNER_CANDIDATES = [
    os.path.join(NEAT_RUNS_DIR, "winner.pkl"),
    os.path.join(NEAT_RUNS_DIR, "neat_best_genome.pkl"),
]

THRESHOLDS_CSV = os.path.join(NEAT_RUNS_DIR, "eval_thresholds.csv")
BEST_TRADES_CSV = os.path.join(NEAT_RUNS_DIR, "eval_best_trades.csv")
PNL_PNG = os.path.join(NEAT_RUNS_DIR, "eval_pnl_best_threshold.png")
THRESHOLD_CURVE_PNG = os.path.join(NEAT_RUNS_DIR, "eval_threshold_curve.png")

NEAT_CONFIG_PATH = os.path.join(ROOT_DIR, "config", "neat_config_encoder.txt")

# === Сетка порогов ============================================================
THRESHOLDS = [
    0.10, 0.12, 0.14, 0.16, 0.18,
    0.20, 0.22, 0.24, 0.26, 0.28,
    0.30, 0.35, 0.40, 0.45, 0.50,
]

# === Загрузка данных ==========================================================

def load_data():
    if not os.path.exists(LATENTS_CACHE):
        raise RuntimeError(f"Latents cache not found: {LATENTS_CACHE}\nЗапусти train_neat.py сначала.")

    z = np.load(LATENTS_CACHE)
    latents = z["LATENTS"]
    target = z["FUTURE_PIPS"]
    print(f"Loaded latents cache: {latents.shape}, FUTURE_PIPS: {target.shape}")
    return latents, target


def load_winner():
    for path in WINNER_CANDIDATES:
        if os.path.exists(path):
            with open(path, "rb") as f:
                g = pickle.load(f)
            print(f"Loaded best genome from: {path}")
            print(f"Best genome fitness (training): {g.fitness:.6f}")
            return g

    raise RuntimeError(
        "Winner genome not found. Tried:\n  " +
        "\n  ".join(WINNER_CANDIDATES)
    )


def build_network(genome):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return net


# === Оценка торговли для заданного порога =====================================

def eval_at_threshold(pred, target, threshold, return_trades=False):
    """
    pred      : np.array shape (N,) — предсказания NEAT (скаляр на бар)
    target    : np.array shape (N,) — FUTURE_PIPS (сколько цена реально прошла)
    threshold : float — |pred| >= threshold -> торгуем
    return_trades : если True — вернуть список сделок (для лучшего порога)
    """
    trades = []
    total_reward = 0.0
    total_bars = len(target)

    for i, (p, m) in enumerate(zip(pred, target)):
        direction = 1 if p >= 0 else -1
        reward = direction * m  # прибыль в пипсах

        if abs(p) >= threshold:
            trades.append({
                "index": i,
                "pred": float(p),
                "future_pips": float(m),
                "direction": float(direction),
                "reward": float(reward),
            })
            total_reward += reward

    if not trades:
        metrics = {
            "trades": 0,
            "winrate": None,
            "avg_reward_trade": None,
            "avg_reward_bar": 0.0,
            "avg_move_correct": None,
            "avg_move_wrong": None,
        }
        return (metrics, trades) if return_trades else (metrics, None)

    rewards = np.array([t["reward"] for t in trades], dtype=np.float32)
    correct_moves = np.array(
        [t["future_pips"] for t in trades if t["reward"] > 0], dtype=np.float32
    )
    wrong_moves = np.array(
        [t["future_pips"] for t in trades if t["reward"] <= 0], dtype=np.float32
    )

    winrate = float((rewards > 0).mean())
    avg_reward_trade = float(total_reward / len(trades))
    avg_reward_bar = float(total_reward / total_bars)
    avg_move_correct = (
        float(correct_moves.mean()) if correct_moves.size > 0 else None
    )
    avg_move_wrong = (
        float(wrong_moves.mean()) if wrong_moves.size > 0 else None
    )

    metrics = {
        "trades": len(trades),
        "winrate": winrate,
        "avg_reward_trade": avg_reward_trade,
        "avg_reward_bar": avg_reward_bar,
        "avg_move_correct": avg_move_correct,
        "avg_move_wrong": avg_move_wrong,
    }

    return (metrics, trades) if return_trades else (metrics, None)


# === Основной скрипт ==========================================================

def main():
    os.makedirs(NEAT_RUNS_DIR, exist_ok=True)

    latents, target = load_data()
    winner = load_winner()
    net = build_network(winner)

    # --- Предсказания модели ---
    raw = np.array([net.activate(x)[0] for x in latents], dtype=np.float32)

    # raw ∈ [0,1] — трактуем как вероятность роста
    proba_up = np.clip(raw, 0.0, 1.0)

    direction = np.where(proba_up >= 0.5, 1.0, -1.0)
    confidence = 2.0 * np.abs(proba_up - 0.5)  # [0, 1]

    # --- Предсказания сети NEAT ---
    # raw_o ∈ [0, 1] — трактуем как вероятность роста (CALL)
    raw = np.array([net.activate(x)[0] for x in latents], dtype=np.float32)

    # Страхуемся: жёстко зажимаем в [0, 1]
    proba_up = np.clip(raw, 0.0, 1.0)

    # Направление и уверенность
    direction = np.where(proba_up >= 0.5, 1.0, -1.0)  # +1 = LONG, -1 = SHORT
    confidence = 2.0 * np.abs(proba_up - 0.5)  # 0..1, 0 — сомнение, 1 — максимум

    # Итоговый сигнал для анализа: в [-1, 1]
    # знак = направление, |pred| = уверенность
    pred = direction * confidence

    print("\n=== Predictions distribution ===")
    print(f"pred min       : {pred.min():.4f}")
    print(f"pred max       : {pred.max():.4f}")
    print(f"pred mean      : {pred.mean():.4f}")
    print(f"pred median    : {np.median(pred):.4f}")
    abs_p = np.abs(pred)
    print(f"|pred| mean    : {abs_p.mean():.4f}")
    print(f"|pred| p50    : {np.percentile(abs_p, 50):.4f}")
    print(f"|pred| p75    : {np.percentile(abs_p, 75):.4f}")
    print(f"|pred| p90    : {np.percentile(abs_p, 90):.4f}")
    print(f"|pred| p95    : {np.percentile(abs_p, 95):.4f}")
    print(f"|pred| p99    : {np.percentile(abs_p, 99):.4f}")
    print(f"Share LONG preds  : {(pred >= 0).mean():.4f}")
    print(f"Share SHORT preds : {(pred < 0).mean():.4f}")

    # --- Прогон по сетке порогов ---
    print("\n=== Threshold grid evaluation ===")

    rows = []
    best_thr = None
    best_reward_bar = -1e9

    for thr in THRESHOLDS:
        metrics, _ = eval_at_threshold(pred, target, thr, return_trades=False)

        print(f"\n--- threshold = {thr:.2f} ---")
        if metrics["trades"] == 0:
            print("No trades at this threshold.")
        else:
            print(f"Trades: {metrics['trades']} / {len(target)}")
            print(f"Winrate            : {metrics['winrate'] * 100:.2f}%")
            print(f"Avg reward/trade   : {metrics['avg_reward_trade']:.6f}")
            print(f"Avg reward/bar     : {metrics['avg_reward_bar']:.6f}")
            print(f"Avg move(correct)  : {metrics['avg_move_correct']}")
            print(f"Avg move(wrong)    : {metrics['avg_move_wrong']}")

        rows.append({
            "threshold": thr,
            "trades": metrics["trades"],
            "winrate": metrics["winrate"],
            "avg_reward_trade": metrics["avg_reward_trade"],
            "avg_reward_bar": metrics["avg_reward_bar"],
            "avg_move_correct": metrics["avg_move_correct"],
            "avg_move_wrong": metrics["avg_move_wrong"],
        })

        if metrics["avg_reward_bar"] is not None and metrics["avg_reward_bar"] > best_reward_bar:
            best_reward_bar = metrics["avg_reward_bar"]
            best_thr = thr

    # --- Сохранение таблицы в CSV (Excel-friendly) ---
    fieldnames = [
        "threshold",
        "trades",
        "winrate",
        "avg_reward_trade",
        "avg_reward_bar",
        "avg_move_correct",
        "avg_move_wrong",
    ]
    with open(THRESHOLDS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved thresholds table to: {THRESHOLDS_CSV}")

    # --- Лучшый порог ---
    if best_thr is None:
        print("\nНет порога с трейдами. Оценка невозможна.")
        return

    print("\n============== BEST THRESHOLD ==============")
    print(f"Best threshold: {best_thr:.2f}")
    print(f"Best avg reward/bar: {best_reward_bar:.6f}")

    # === Детальный анализ лучшего порога ======================================
    best_metrics, best_trades = eval_at_threshold(
        pred, target, best_thr, return_trades=True
    )

    print("\n=== Detailed stats for best threshold ===")
    print(f"Trades           : {best_metrics['trades']} / {len(target)}")
    print(f"Winrate          : {best_metrics['winrate'] * 100:.2f}%")
    print(f"Avg reward/trade : {best_metrics['avg_reward_trade']:.6f}")
    print(f"Avg reward/bar   : {best_metrics['avg_reward_bar']:.6f}")
    print(f"Avg move(correct): {best_metrics['avg_move_correct']}")
    print(f"Avg move(wrong)  : {best_metrics['avg_move_wrong']}")

    # --- Строим PnL по сделкам (по порядку появления) ---
    rewards = np.array([t["reward"] for t in best_trades], dtype=np.float32)
    cum_pnl = rewards.cumsum()

    # Сохраним сделки с PnL для Excel
    with open(BEST_TRADES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "pred", "future_pips", "direction", "reward", "cum_pnl"])
        running = 0.0
        for t in best_trades:
            running += t["reward"]
            writer.writerow([
                t["index"],
                t["pred"],
                t["future_pips"],
                t["direction"],
                t["reward"],
                running,
            ])

    print(f"Saved best trades with PnL to: {BEST_TRADES_CSV}")

    # --- График PnL по трейдам ---
    plt.figure(figsize=(10, 5))
    plt.plot(cum_pnl)
    plt.title(f"PNL by trades (threshold={best_thr:.2f})")
    plt.xlabel("Trade index")
    plt.ylabel("Cumulative PnL (pips)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PNL_PNG, dpi=150)
    # plt.show()  # можно раскомментировать, если хочешь увидеть в окне
    print(f"Saved PnL plot to: {PNL_PNG}")

    # --- Кривая threshold -> avg_reward/bar ---
    thr_vals = [r["threshold"] for r in rows]
    rew_vals = [r["avg_reward_bar"] if r["avg_reward_bar"] is not None else 0.0 for r in rows]

    plt.figure(figsize=(8, 4))
    plt.plot(thr_vals, rew_vals, marker="o")
    plt.axvline(best_thr, color="red", linestyle="--", alpha=0.7, label=f"best={best_thr:.2f}")
    plt.title("Threshold vs Avg reward per bar")
    plt.xlabel("Threshold |pred|")
    plt.ylabel("Avg reward per bar (pips)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(THRESHOLD_CURVE_PNG, dpi=150)
    # plt.show()
    print(f"Saved threshold curve plot to: {THRESHOLD_CURVE_PNG}")

    # --- Топ-20 лучших и худших сделок ---
    sorted_by_reward = sorted(best_trades, key=lambda t: t["reward"], reverse=True)
    best_20 = sorted_by_reward[:20]
    worst_20 = sorted_by_reward[-20:]

    print("\n=== Top 20 BEST trades (by reward) ===")
    for t in best_20:
        print(
            f"idx={t['index']:6d}  pred={t['pred']:+.4f}  "
            f"future_pips={t['future_pips']:+.5f}  reward={t['reward']:+.5f}"
        )

    print("\n=== Top 20 WORST trades (by reward) ===")
    for t in worst_20:
        print(
            f"idx={t['index']:6d}  pred={t['pred']:+.4f}  "
            f"future_pips={t['future_pips']:+.5f}  reward={t['reward']:+.5f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
