"""
tune_lagrangian_eta_oversight.py
------------------------------------
Tunes eta (the Lagrange multiplier's dual-ascent learning rate) for
Architecture D on the oversight signal. Mirrors this project's
established tuning discipline exactly (tune_hyperparams_oversight.py,
tune_hyperparams_battery.py): VALIDATION_SEEDS disjoint from the real
sweep's SEEDS (0-19), VALIDATION_TIMESTEPS raised to 200,000 from the
start (not re-discovered the hard way a third time -- both prior
tuners needed this same fix after an initial too-small-budget attempt;
applying it proactively here).

d_threshold is NOT tuned here -- it's a confirmed design decision
(0.05), not a free hyperparameter to search; see lagrangian_ppo.py's
module docstring for why. Only eta and (secondarily) log_std_init/
learning_rate/gamma are searched, mirroring the original two tuners'
two-phase structure (base RL hyperparameters first, then the
experiment-specific knob -- here eta, there lambda_*_a).

Scoring: uses this project's established anti-gaming pattern (a
MIN_SUCCESS_RATE floor disqualifies a frozen/inert policy, which would
otherwise trivially satisfy the constraint by never moving -- see
tune_hyperparams_oversight.py's _score docstring for the original
incident this guards against), PLUS a constraint-specific penalty:
heavily penalizes cost_rate exceeding d_threshold (eta too small to
enforce the constraint), with a mild penalty for being needlessly far
UNDER threshold too (eta too large, over-correcting into excess
caution and wasted success_rate).

Usage:
    python tune_lagrangian_eta_oversight.py

Expected runtime: rough estimate only -- scales with
    len(LR_CANDIDATES) x len(LOG_STD_INIT_CANDIDATES) x len(VALIDATION_SEEDS)
  + len(ETA_CANDIDATES) x len(VALIDATION_SEEDS)
separate training runs at VALIDATION_TIMESTEPS each. Run time_probe.py
(or reuse its earlier measurements) before trusting a guess -- per this
project's standing burned-once-already discipline on hand-derived time
estimates.
"""

import json
import os

import numpy as np

from env_oversight import ContinuousNavOversightEnv
from lagrangian_ppo import train_lagrangian_ppo, _quick_rollout_for_lambda

RESULTS_DIR = "results"
HYPERPARAMS_PATH = f"{RESULTS_DIR}/hyperparams_lagrangian_oversight.json"

VALIDATION_NET_SIZE = [64]
VALIDATION_TIMESTEPS = 200_000  # raised from the start -- see module docstring
VALIDATION_SEEDS = [100, 101, 102]  # disjoint from the real sweep's SEEDS (0-19)
UPDATE_INTERVAL = 10_000  # matches the real sweep's default cadence for lambda updates

D_THRESHOLD = 0.05  # confirmed design decision, not searched -- see module docstring

LR_CANDIDATES = [1e-3, 3e-4]
GAMMA = 0.99  # fixed, matching both prior tuners' own converged choice for this task family
LOG_STD_INIT_CANDIDATES = [-1.0, -0.5, 0.0]
ETA_CANDIDATES = [0.01, 0.05, 0.1, 0.5, 1.0]  # log-ish spacing -- covers "barely
                                                # enforces the constraint" through
                                                # "overcorrects aggressively"


def _score(row, d_threshold=D_THRESHOLD):
    """
    See module docstring. MIN_SUCCESS_RATE floor catches a frozen/inert
    policy (cost_rate~0 "trivially" satisfies any constraint by never
    moving, but that's not a useful baseline). Among candidates that
    pass the floor: heavily penalize exceeding the threshold (eta too
    weak to enforce it), mildly penalize being far under it (eta too
    strong, wasting achievable success_rate on needless caution).
    """
    MIN_SUCCESS_RATE = 0.05
    if row["success_rate"] < MIN_SUCCESS_RATE:
        return -1.0
    over_penalty = max(0.0, row["cost_rate"] - d_threshold) * 5.0
    distance_penalty = abs(row["cost_rate"] - d_threshold) * 1.0
    return 3.0 * row["success_rate"] - over_penalty - distance_penalty


def _evaluate_d(net_size, timesteps, seed, hp, eta):
    model, history = train_lagrangian_ppo(
        ContinuousNavOversightEnv, cost_key="violated", signal_key="r_oversight",
        net_size=net_size, timesteps=timesteps, seed=seed, hp=hp,
        d_threshold=D_THRESHOLD, eta=eta, update_interval=UPDATE_INTERVAL,
        return_history=True,
    )
    # Final validation metrics, on a FRESH rollout batch (not the internal
    # dual-ascent rollouts used during training, to avoid any optimistic
    # bias from repeatedly evaluating on the same seed block).
    final = _quick_rollout_for_lambda(
        model, lambda: ContinuousNavOversightEnv(), cost_key="violated",
        n_episodes=50, seed=seed * 1000 + 500_000,
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
            "raise VALIDATION_TIMESTEPS rather than trusting a result chosen this way "
            "(mirrors both prior tuners' own fix for this exact failure mode)."
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
    print("\nNOTE: this does not tune update_interval (fixed at "
          f"{UPDATE_INTERVAL} steps, matching the real sweep's default) or "
          "lambda_init (fixed at 0.0, an unbiased starting point) -- both are "
          "confirmed design decisions, not free hyperparameters, per this "
          "project's design discussion.")
