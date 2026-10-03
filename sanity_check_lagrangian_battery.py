"""
sanity_check_lagrangian_battery.py
--------------------------------------
Mirrors sanity_check_lagrangian_oversight.py's role and structure, with
two battery-specific choices made deliberately, not by default:

1. Checks at CHECK_TIMESTEPS=10,000 -- the real sweep's SMALLEST
   budget, not 200,000 like the oversight version. Found during this
   project's own tuning: at 200,000 steps, battery's standard AND
   override_w8 conditions are both nearly solved by EVERY eta
   candidate (success_rate~1.0, cost_rate~0.0) -- a ceiling effect that
   makes 200k uninformative for checking whether the tuned eta actually
   matters. The real sweep's smallest cell (10,000 steps) is the
   budget most likely to still show real eta-sensitivity, since it's
   furthest from that ceiling.

2. Checks under CHECK_ADVERSARIAL_WEIGHT=8 (override_w8), not the
   standard/no-conflict condition -- found during tuning that
   adversarial_weight=0 is trivially easy for this environment (even g
   ALONE, with no battery-awareness, already gets crash_rate=0.00 per
   config_battery.py's own notes), so it doesn't stress the constraint
   mechanism at all. override_w8 is where the real sweep's own
   headline comparisons are drawn (§7.3/§7.6's "at the highest conflict
   weight tested"), and where eta actually needs to do work.

Uses n_updates (not update_interval) for the dual-ascent schedule --
this MUST match what run_experiment_lagrangian_battery.py actually uses
via train_architecture_d's n_updates=N_DUAL_UPDATES=20, since at a
small budget like 10,000 steps, a fixed update_interval=10,000 would
fire only ONE lambda update total (uninformative -- see
run_experiment_lagrangian_oversight.py's n_updates docstring for the
full reasoning this mirrors).

Usage:
    python sanity_check_lagrangian_battery.py

Requires: the TUNED hyperparameters from tune_lagrangian_eta_battery.py
(results/hyperparams_lagrangian_battery.json) -- falls back to a
placeholder with a warning if not found.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from env_battery_continuous import ContinuousNavBatteryEnv
from lagrangian_ppo import train_lagrangian_ppo, _quick_rollout_for_lambda

CHECK_TIMESTEPS = 10_000
N_UPDATES = 20  # matches run_experiment_lagrangian_battery.py's N_DUAL_UPDATES exactly
CHECK_ADVERSARIAL_WEIGHT = 8  # override_w8 -- see module docstring for why not 0
CHECK_NET_SIZE = [64]
CHECK_SEED = 0
N_EVAL_EPISODES_FINAL = 100

HP_PATH = "results/hyperparams_lagrangian_battery.json"


if __name__ == "__main__":
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            hp = json.load(f)
        eta = hp.get("lagrangian_eta", 0.1)
        d_threshold = hp.get("lagrangian_d_threshold", 0.05)
        print(f"NOTE: loaded TUNED hyperparameters from {HP_PATH}: {hp}\n")
    else:
        hp = {"learning_rate": 3e-4, "gamma": 0.99, "log_std_init": -0.5}
        eta, d_threshold = 0.1, 0.05
        print(f"NOTE: {HP_PATH} not found -- using placeholder hyperparameters "
              f"{hp}, eta={eta}, d_threshold={d_threshold}. Run "
              f"tune_lagrangian_eta_battery.py first for tuned values.\n")

    print(f"Training Architecture D (battery) for {CHECK_TIMESTEPS} steps "
          f"(the real sweep's SMALLEST budget), under override_w{CHECK_ADVERSARIAL_WEIGHT} "
          f"(NOT the easy standard condition), {N_UPDATES} lambda updates, "
          f"target cost_rate={d_threshold}...\n")

    model, history = train_lagrangian_ppo(
        ContinuousNavBatteryEnv, cost_key="crashed", signal_key="r_battery",
        net_size=CHECK_NET_SIZE, timesteps=CHECK_TIMESTEPS, seed=CHECK_SEED, hp=hp,
        adversarial_weight=CHECK_ADVERSARIAL_WEIGHT,
        d_threshold=d_threshold, eta=eta, n_updates=N_UPDATES,
        return_history=True,
    )

    for row in history:
        print(f"  steps={row['steps']:>7} cost_rate={row['cost_rate']:.3f} "
              f"success_rate={row['success_rate']:.3f} lambda={row['lambda']:.4f}")

    final = _quick_rollout_for_lambda(
        model, lambda: ContinuousNavBatteryEnv(adversarial_weight=CHECK_ADVERSARIAL_WEIGHT),
        cost_key="crashed", n_episodes=N_EVAL_EPISODES_FINAL, seed=999_000,
    )
    print(f"\nFinal held-out evaluation ({N_EVAL_EPISODES_FINAL} episodes, "
          f"override_w{CHECK_ADVERSARIAL_WEIGHT}): success_rate={final['success_rate']:.3f} "
          f"cost_rate={final['cost_rate']:.3f} (target {d_threshold})")

    last_third = history[-max(1, len(history) // 3):]
    lambdas = [r["lambda"] for r in last_third]
    lam_range = max(lambdas) - min(lambdas)
    if lam_range > 0.5 * (max(lambdas) if max(lambdas) > 0 else 1.0):
        print(f"\n  WARNING: lambda is still swinging widely in the final third "
              f"(range={lam_range:.3f} over values {lambdas}) -- consider a smaller "
              f"eta, or treat this budget's D results with caution.")
    else:
        print(f"\n  Lambda looks reasonably converged in the final third "
              f"(range={lam_range:.3f}).")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = [r["steps"] for r in history]
    axes[0].plot(steps, [r["success_rate"] for r in history], marker="o")
    axes[0].set_xlabel("training steps"); axes[0].set_ylabel("success_rate")
    axes[0].set_title("Does success_rate climb?")

    axes[1].plot(steps, [r["cost_rate"] for r in history], marker="o", color="tab:red")
    axes[1].axhline(d_threshold, color="gray", linestyle="--", label=f"target ({d_threshold})")
    axes[1].set_xlabel("training steps"); axes[1].set_ylabel("cost_rate (crash_rate)")
    axes[1].set_title(f"Does cost_rate settle near target? (override_w{CHECK_ADVERSARIAL_WEIGHT})")
    axes[1].legend()

    axes[2].plot(steps, [r["lambda"] for r in history], marker="o", color="tab:purple")
    axes[2].set_xlabel("training steps"); axes[2].set_ylabel("lambda")
    axes[2].set_title("Does lambda converge (not oscillate/diverge)?")

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sanity_check_lagrangian_battery_learning_curves.png", dpi=150)
    print("\nSaved results/sanity_check_lagrangian_battery_learning_curves.png")
