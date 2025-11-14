# bin/train_neat.py
"""
Тренування NEAT-агентів поверх латентів енкодера Encoder1DCNN_LSTM.

Пайплайн:
1) Беремо перший shard з out_shards/dataset_part_0000.npz
   Очікувані ключі: "X", "y_bin", "y_cont".
   - X:      (N, 96, 34)
   - y_cont: (N,) сирий безмасштабний таргет (типу (close/open - 1))

2) Проганяємо X через encoder_best.pt -> LATENTS (N, latent_dim)

3) Формуємо таргет для NEAT:
       FUTURE_PIPS = y_cont * 10000.0
   (тобто беремо безрозмірну дохідність і переводимо в "піпси")

4) Зберігаємо latents_cache.npz

5) Запускаємо NEAT з 2 виходами:
       out_long, out_short in R, pred = sigmoid(out_long) - sigmoid(out_short)

6) Fitness:
   - базовий reward за напрямок + силу руху в піпсах;
   - штраф за "колапс": всі сигнали в один бік / дуже мала дисперсія.
"""

import os
import sys
import multiprocessing as mp
import pickle
import numpy as np
import torch
import neat

# === Шляхи так само, як у train_encoder.py ===================================
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CUR_DIR)

SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import Encoder1DCNN_LSTM  # той самий енкодер, що й у train_encoder.py

# === Константи для NEAT ======================================================
SHARDS_DIR      = os.path.join(ROOT_DIR, "out_shards")
CHK_ENCODER     = os.path.join(ROOT_DIR, "checkpoints", "encoder_best.pt")
NEAT_CONFIG     = os.path.join(ROOT_DIR, "config", "neat_config_encoder.txt")
NEAT_RUNS_DIR   = os.path.join(ROOT_DIR, "neat_runs")
LATENTS_CACHE   = os.path.join(NEAT_RUNS_DIR, "latents_cache.npz")
BEST_GENOME_PKL = os.path.join(NEAT_RUNS_DIR, "neat_best_genome.pkl")
# ===== Fitness hyperparams ("trading standard") =====
# Сколько окон берем для оценки каждого генома
SUBSAMPLE_N = 50000       # если данных меньше — возьмем всё

# Порог "уверенности" для торговли во время обучения
TRAIN_THRESHOLD = 0.2     # |pred| < 0.2 => пропускаем бар (no trade)

# Coverage: доля баров, где есть сделка
MIN_COVERAGE_HARD = 0.01   # 0.5% — ниже этого сразу плохой фитнес
COVERAGE_TARGET   = 0.05    # целевая доля сделок: 2% от SUBSAMPLE_N

# Сколько первых сделок считаем "разогревом" и не учитываем в шарпе
WARMUP_TRADES = 500

# Штрафы
W_COVERAGE        = 3.0   # штраф за недобор по coverage
W_IMBALANCE       = 0.5   # штраф за сильный дисбаланс LONG/SHORT
STD_MIN           = 0.05  # минимальная разумная дисперсия предсказаний
PENALTY_SMALL_STD = 0.3   # штраф, если std(pred) слишком маленький

os.makedirs(NEAT_RUNS_DIR, exist_ok=True)

LATENTS = None       # (N, latent_dim)
FUTURE_PIPS = None   # (N,)

LATENT_DIM = 128          # має збігатися з CONFIG["latent_dim"] у train_encoder.py
NEAT_GENERATIONS = 30
MAX_SAMPLES_FOR_NEAT = 50000


# === Крок 1: Предобчислення латентів =========================================
def precompute_latents(force_recompute: bool = False):
    """
    Предвычисляем латенты энкодера и future_move_pips для одного NPZ-шарда
    и сохраняем весь массив в latents_cache.npz.

    Если force_recompute == False и кеш уже есть — просто загружаем его.
    """
    global LATENTS, FUTURE_PIPS

    # Если кеш уже есть и не просили пересчитать — просто загрузим
    if (not force_recompute) and os.path.exists(LATENTS_CACHE):
        print(f"Using existing latents cache: {LATENTS_CACHE}")
        with np.load(LATENTS_CACHE) as z:
            LATENTS = z["LATENTS"]
            FUTURE_PIPS = z["FUTURE_PIPS"]
        print(f"Loaded LATENTS: {LATENTS.shape}, FUTURE_PIPS: {FUTURE_PIPS.shape}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device for encoder (precompute latents): {device}")

    # Берём первый шард из SHARDS_DIR
    shard_files = sorted(
        [
            os.path.join(SHARDS_DIR, p)
            for p in os.listdir(SHARDS_DIR)
            if p.startswith("dataset_part_") and p.endswith(".npz")
        ]
    )
    if not shard_files:
        raise RuntimeError(f"No shard files found in {SHARDS_DIR}")
    shard_path = shard_files[0]
    print(f"Using shard for NEAT data: {shard_path}")

    # Загружаем X и таргет
    with np.load(shard_path) as z:
        X = z["X"]          # (N, 96, 34)
        y_cont = z["y_cont"]  # (N,)
    N_total = X.shape[0]
    print(f"Loaded {N_total} samples from shard, keeping ALL {N_total} for NEAT cache")

    # Строим энкодер и грузим веса
    encoder = Encoder1DCNN_LSTM(in_channels=34, latent_dim=LATENT_DIM).to(device)
    state_dict = torch.load(CHK_ENCODER, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    # Прогоняем весь X через энкодер батчами
    batch_size = 512
    latents_list = []
    with torch.no_grad():
        for start in range(0, N_total, batch_size):
            end = min(start + batch_size, N_total)
            x_batch = torch.from_numpy(X[start:end]).to(device)
            z = encoder(x_batch)          # (B, latent_dim)
            latents_list.append(z.cpu().numpy())
    LATENTS = np.concatenate(latents_list, axis=0)   # (N_total, 128)

    # Таргет: future_move_pips.
    # Здесь предполагаем, что y_cont уже сохранён как (Close(t+1) - Close(t)) * 10000.
    # Если это не так — формулу можно скорректировать под твою схему в make_dataset.
    FUTURE_PIPS = y_cont.astype(np.float32)

    print(f"Precomputed latents: {LATENTS.shape}")
    print(f"Targets (future_move_pips): {FUTURE_PIPS.shape}")

    os.makedirs(NEAT_RUNS_DIR, exist_ok=True)
    np.savez_compressed(LATENTS_CACHE, LATENTS=LATENTS, FUTURE_PIPS=FUTURE_PIPS)
    print(f"Saved latents cache to: {LATENTS_CACHE}")




# === Крок 2: Reward-функція для одного кроку =================================
def compute_step_reward(
    pred: float,
    future_pips: float,
    threshold: float = 0.2,
    max_pips: float = 10.0,
    w_dir: float = 1.0,
    w_amp: float = 1.0,
    w_trade: float = 0.1,
):
    """
    pred        : сигнал агента (long-short), >0 = CALL, <0 = PUT
    future_pips : фактичний рух у піпсах
    threshold   : поріг впевненості для входу
    """
    if abs(pred) <= threshold:
        return 0.0, False, 0.0, None

    action = "CALL" if pred > 0 else "PUT"

    direction_correct = (
        (action == "CALL" and future_pips > 0) or
        (action == "PUT"  and future_pips < 0)
    )
    sign_correct = 1.0 if direction_correct else -1.0

    strength = min(abs(future_pips), max_pips) / max_pips  # 0..1

    reward = (
        w_dir * sign_correct +
        w_amp * sign_correct * strength -
        w_trade
    )
    return reward, True, abs(future_pips), direction_correct


# === Крок 3: Оцінка геному (fitness) =========================================
def eval_genome(genome, config):
    """
    Оценка одного генома по "трейдинговому стандарту":

    - работаем на LATENTS / FUTURE_PIPS из latents_cache.npz
    - для каждого генома берём случайную подвыборку SUBSAMPLE_N окон
    - торгуем, только если |pred| >= TRAIN_THRESHOLD
    - первые WARMUP_TRADES сделок считаем "прогревом" и в метриках не учитываем
    - основная метрика: Sharpe-подобный коэффициент = mean(reward) / std(reward)
    - штрафы:
        * за низкий coverage (мало сделок)
        * за дисбаланс LONG/SHORT
        * за слишком маленькую дисперсию предсказаний (слишком "плоская" стратегия)
    """

    global LATENTS, FUTURE_PIPS

    # Ленивая загрузка кеша в каждом воркере
    if LATENTS is None or FUTURE_PIPS is None:
        if not os.path.exists(LATENTS_CACHE):
            raise RuntimeError(
                f"Latents cache not found in worker process: {LATENTS_CACHE}"
            )
        with np.load(LATENTS_CACHE) as z:
            LATENTS = z["LATENTS"]
            FUTURE_PIPS = z["FUTURE_PIPS"]

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    N = LATENTS.shape[0]
    if N == 0:
        return -1.0

    # --- динамическая подвыборка из всех латентов ---
    subsample_n = min(SUBSAMPLE_N, N)
    if subsample_n < 1000:
        # слишком мало данных для адекватной оценки
        return -1.0

    if N > subsample_n:
        idx = np.random.choice(N, size=subsample_n, replace=False)
    else:
        idx = np.arange(N)

    rewards = []
    dirs = []        # +1 (long) или -1 (short)
    preds_used = []  # предсказания, по которым реально входили в сделку

    total_trades = 0

    for i in idx:
        x = LATENTS[i]
        future_pips = float(FUTURE_PIPS[i])

        out = net.activate(x.tolist())
        if len(out) != 2:
            # архитектура NEAT-сети нарушена
            return -1.0

        out_long, out_short = out
        long_p = 1.0 / (1.0 + np.exp(-out_long))
        short_p = 1.0 / (1.0 + np.exp(-out_short))
        pred = long_p - short_p  # >0 => long, <0 => short

        # Фильтр по уверенности: слишком малые |pred| = "нет сделки"
        if abs(pred) < TRAIN_THRESHOLD:
            continue

        direction = 1.0 if pred > 0 else -1.0
        reward = direction * future_pips  # будущий ход в пипсах, с учётом направления

        rewards.append(reward)
        dirs.append(direction)
        preds_used.append(pred)
        total_trades += 1

    # --- Жесткие проверки против "снайпера" ---
    if total_trades <= WARMUP_TRADES:
        # Почти не торгует (и все сделки могут быть случайным успехом)
        return -1.0

    coverage = total_trades / float(subsample_n)
    if coverage < MIN_COVERAGE_HARD:
        # Совсем мало сделок относительно выборки
        return -1.0

    # --- Отбрасываем первые сделки как разогрев ---
    eff_rewards = np.array(rewards[WARMUP_TRADES:], dtype=np.float32)
    eff_dirs = np.array(dirs[WARMUP_TRADES:], dtype=np.float32)
    eff_preds = np.array(preds_used[WARMUP_TRADES:], dtype=np.float32)

    if eff_rewards.size == 0:
        return -1.0

    mean_r = float(eff_rewards.mean())
    std_r = float(eff_rewards.std())
    if not np.isfinite(std_r) or std_r < 1e-6:
        sharpe = mean_r / 1e-6
    else:
        sharpe = mean_r / (std_r + 1e-6)

    # Winrate (дополнительный маленький плюс за устойчивую направленность)
    wins = float((eff_rewards > 0).mean())

    # Базовый fitness: Sharpe + чуть-чуть winrate
    fitness = sharpe + 0.1 * (wins - 0.5)

    # --- Штраф за недобор по coverage ---
    cov_penalty = max(0.0, COVERAGE_TARGET - coverage)
    fitness -= W_COVERAGE * cov_penalty

    # --- Штраф за дисбаланс LONG / SHORT ---
    longs_share = float((eff_dirs > 0).mean())
    shorts_share = float((eff_dirs < 0).mean())
    imbalance = abs(longs_share - shorts_share)
    fitness -= W_IMBALANCE * imbalance

    # --- Штраф за слишком маленькую дисперсию предсказаний ---
    std_pred = float(eff_preds.std())
    if std_pred < STD_MIN:
        fitness -= PENALTY_SMALL_STD

    # На всякий случай обрезаем совсем безумные случаи
    if not np.isfinite(fitness):
        return -1.0

    return float(fitness)



# === Крок 4: Запуск NEAT ======================================================
def run_neat():
    global LATENTS, FUTURE_PIPS

    precompute_latents(force_recompute=True)

    print("Loading NEAT config from:", NEAT_CONFIG)
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG,
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(
        neat.Checkpointer(
            10,
            filename_prefix=os.path.join(NEAT_RUNS_DIR, "neat-checkpoint-")
        )
    )

    num_workers = mp.cpu_count()
    print(f"Using ParallelEvaluator with {num_workers} workers")
    pe = neat.ParallelEvaluator(num_workers, eval_genome)

    winner = pop.run(pe.evaluate, NEAT_GENERATIONS)

    print("\n=== NEAT finished ===")
    print("Best fitness:", winner.fitness)

    with open(BEST_GENOME_PKL, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved best genome to: {BEST_GENOME_PKL}")


def main():
    run_neat()


if __name__ == "__main__":
    main()
