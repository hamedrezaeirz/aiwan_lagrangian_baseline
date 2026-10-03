"""
tune_lagrangian_eta_battery.py
----------------------------------
Tunes eta (the Lagrange multiplier's dual-ascent learning rate) for
Architecture D on the battery signal. Mirrors
tune_lagrangian_eta_oversight.py exactly, with cost_key="crashed" and
signal_key="r_battery" in place of "violated"/"r_oversight", and
VALIDATION_SEEDS/SEEDS drawn from config_battery.py (10 real seeds, not
20 -- battery is a robustness check, not the primary result, same
reasoning as the rest of this project's battery-vs-oversight seed-count
split).

d_threshold is NOT tuned here -- confirmed design decision (0.05),
shared with the oversight version. See lagrangian_ppo.py's module
docstring.

Usage:
    python tune_lagrangian_eta_battery.py

Expected runtime: same order of magnitude as
tune_lagrangian_eta_oversight.py's run on this machine -- scales with
    len(LR_CANDIDATES) x len(LOG_STD_INIT_CANDIDATES) x len(VALIDATION_SEEDS)
  + len(ETA_CANDIDATES) x len(VALIDATION_SEEDS)
separate training runs at VALIDATION_TIMESTEPS each.
"""

import json
import os

import numpy as np

from env_battery_continuous import ContinuousNavBatteryEnv
from lagrangian_ppo import train_lagrangian_ppo, _quick_rollout_for_lambda
from config_battery import VALIDATION_SEEDS

RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams_lagrangian_battery.json"

VALIDATION_NET_SIZE = [64]
VALIDATION_TIMESTEPS = 200_000  # matches the oversight tuner -- same
                                 # "50k was too small the first two
                                 # times" lesson applied proactively.
UPDATE_INTERVAL = 10_000

D_THRESHOLD = 0.05  # confirmed design decision, shared with oversight -- not searched.

TUNING_ADVERSARIAL_WEIGHT = 8  # NOT 0. Found during validation: at
                                # adversarial_weight=0 (no override pressure),
                                # this environment is trivially easy -- even g
                                # ALONE, with no battery-awareness at all,
                                # already achieves crash_rate=0.00 (see
                                # config_battery.py's own sanity-check notes).
                                # Tuning at adversarial_weight=0 produced a
                                # ceiling effect: every eta candidate scored
                                # ~2.95 (success_rate=1.0, cost_rate~0.0),
                                # indistinguishable from each other -- the
                                # validation never stressed the constraint
                                # mechanism enough to reveal real differences.
                                # Tuning at the real sweep's HARDEST override
                                # weight instead (matching where §7.3/§7.6's
                                # own headline comparisons are drawn, "at the
                                # highest conflict weight tested") is where
                                # eta actually needs to do work, mirroring
                                # tune_hyperparams_oversight.py's own fairness
                                # reasoning for tuning on the harder task.

LR_CANDIDATES = [1e-3, 3e-4]
GAMMA = 0.99
LOG_STD_INIT_CANDIDATES = [-1.0, -0.5, 0.0]
ETA_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0]


def _score(row, d_threshold=D_THRESHOLD):
    """Identical to tune_lagrangian_eta_oversight.py's _score -- see
    that file's docstring for the full rationale (anti-gaming floor +
    constraint-violation penalty + mild over-caution penalty)."""
    MIN_SUCCESS_RATE = 0.05
    if row["success_rate"] < MIN_SUCCESS_RATE:
        return -1.0
    over_penalty = max(0.0, row["cost_rate"] - d_threshold) * 5.0
    distance_penalty = abs(row["cost_rate"] - d_threshold) * 1.0
    return 3.0 * row["success_rate"] - over_penalty - distance_penalty


def _evaluate_d(net_size, timesteps, seed, hp, eta):
    model, history = train_lagrangian_ppo(
        ContinuousNavBatteryEnv, cost_key="crashed", signal_key="r_battery",
        net_size=net_size, timesteps=timesteps, seed=seed, hp=hp,
        adversarial_weight=TUNING_ADVERSARIAL_WEIGHT,
        d_threshold=D_THRESHOLD, eta=eta, update_interval=UPDATE_INTERVAL,
        return_history=True,
    )
    final = _quick_rollout_for_lambda(
        model, lambda: ContinuousNavBatteryEnv(adversarial_weight=TUNING_ADVERSARIAL_WEIGHT),
        cost_key="crashed", n_episodes=50, seed=seed * 1000 + 500_000,
    )
    return final, history


def _mean_score_over_seeds(build_and_eval_fn):
    scores = []
    for s in VALIDATION_SEEDS:
        metrics, _ = build_and_eval_fn(s)
        scores.append(_score(metrics))
    return float(np.mean(scores))


def tune_lr_logstd():
    print("Tuning learning_rate / log_std_init for Architecture D (gamma fixed at "
          f"{GAMMA}, eta fixed at a mid-range placeholder of 0.1 for this phase)...")
    best_score, best_hp = -np.inf, None
    for lr in LR_CANDIDATES:
        for log_std_init in LOG_STD_INIT_CANDIDATES:
            hp = {"learning_rate": lr, "gamma": GAMMA, "log_std_init": log_std_init}

            def build_and_eval(seed, hp=hp):
                return _evaluate_d(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, hp, eta=0.1)

            score = _mean_score_over_seeds(build_and_eval)
            print(f"  lr={lr} log_std_init={log_std_init} -> score={score:.3f}")
            if score > best_score:
                best_score, best_hp = score, {"learning_rate": lr, "gamma": GAMMA,
                                                "log_std_init": log_std_init}
    if best_hp is None or best_score <= -1.0:
        raise SystemExit(
            "Every candidate was disqualified by _score's MIN_SUCCESS_RATE floor -- "
            "raise VALIDATION_TIMESTEPS rather than trusting a result chosen this way."
        )
    print(f"Chosen: {best_hp} (score={best_score:.3f})")
    return best_hp


def tune_eta(base_hp):
    print("Tuning eta (Lagrange multiplier dual-ascent learning rate)...")
    best_score, best_eta = -np.inf, None
    rows_by_eta = {}
    for eta in ETA_CANDIDATES:
        def build_and_eval(seed, eta=eta):
            return _evaluate_d(VALIDATION_NET_SIZE, VALIDATION_TIMESTEPS, seed, base_hp, eta=eta)

        metrics_per_seed = [build_and_eval(s)[0] for s in VALIDATION_SEEDS]
        score = float(np.mean([_score(m) for m in metrics_per_seed]))
        mean_cost = float(np.mean([m["cost_rate"] for m in metrics_per_seed]))
        mean_succ = float(np.mean([m["success_rate"] for m in metrics_per_seed]))
        rows_by_eta[eta] = (mean_succ, mean_cost)
        print(f"  eta={eta:<5} -> score={score:.3f}  "
              f"(mean success_rate={mean_succ:.3f}, mean cost_rate={mean_cost:.3f}, "
              f"target={D_THRESHOLD})")
        if score > best_score:
            best_score, best_eta = score, eta

    if best_eta is None or best_score <= -1.0:
        raise SystemExit(
            "Every eta candidate was disqualified -- check base_hp's own validity "
            "(from tune_lr_logstd) before re-running this search."
        )
    print(f"Chosen: eta={best_eta} (score={best_score:.3f})")
    return best_eta, rows_by_eta


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    base_hp = tune_lr_logstd()
    eta, rows_by_eta = tune_eta(base_hp)

    final_hp = {**base_hp, "lagrangian_eta": eta, "lagrangian_d_threshold": D_THRESHOLD}
    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(final_hp, f, indent=2)
    print(f"\nSaved to {HYPERPARAMS_PATH}: {final_hp}")
