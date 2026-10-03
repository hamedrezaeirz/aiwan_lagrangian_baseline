"""
sanity_check_lagrangian_oversight.py
----------------------------------------
Run this BEFORE committing to a full Architecture D sweep -- mirrors
this project's standing discipline (sanity_check_oversight.py,
sanity_check_battery.py): never trust a new training pipeline at a
real budget without first watching its learning curve.

What's DIFFERENT to check here, specific to Lagrangian-PPO, beyond the
usual "does success_rate climb / does violation_rate drop": does
LAMBDA converge to a stable value, or does it oscillate wildly /
diverge? Dual ascent is a genuinely different failure mode from
anything A/B/C's training could exhibit -- a badly-tuned eta can make
lambda grow without bound (if the policy can't reduce cost_rate fast
enough to catch up) or oscillate (if eta is too large relative to how
quickly the policy responds to a changed lambda). This should be
checked visually, not just inferred from a good final cost_rate.

Usage:
    python sanity_check_lagrangian_oversight.py

Requires: the TUNED hyperparameters from
tune_lagrangian_eta_oversight.py (results/hyperparams_lagrangian_oversight.json)
-- falls back to a placeholder with a warning if not found, exactly
like the original sanity-check scripts' own pattern.
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from env_oversight import ContinuousNavOversightEnv
from lagrangian_ppo import train_lagrangian_ppo, _quick_rollout_for_lambda

CHECK_TIMESTEPS = 500_000
UPDATE_INTERVAL = 10_000
CHECK_NET_SIZE = [2048]
CHECK_SEED = 0
N_EVAL_EPISODES_FINAL = 100

HP_PATH = "results/hyperparams_lagrangian_oversight.json"


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
              f"tune_lagrangian_eta_oversight.py first for tuned values.\n")

    print(f"Training Architecture D (oversight) for {CHECK_TIMESTEPS} steps, "
          f"updating lambda every {UPDATE_INTERVAL} steps, target cost_rate="
          f"{d_threshold}...\n")

    model, history = train_lagrangian_ppo(
        ContinuousNavOversightEnv, cost_key="violated", signal_key="r_oversight",
        net_size=CHECK_NET_SIZE, timesteps=CHECK_TIMESTEPS, seed=CHECK_SEED, hp=hp,
        d_threshold=d_threshold, eta=eta, update_interval=UPDATE_INTERVAL,
        return_history=True,
    )

    for row in history:
        print(f"  steps={row['steps']:>7} cost_rate={row['cost_rate']:.3f} "
              f"success_rate={row['success_rate']:.3f} lambda={row['lambda']:.4f}")

    final = _quick_rollout_for_lambda(
        model, lambda: ContinuousNavOversightEnv(), cost_key="violated",
        n_episodes=N_EVAL_EPISODES_FINAL, seed=999_000,
    )
    print(f"\nFinal held-out evaluation ({N_EVAL_EPISODES_FINAL} episodes): "
          f"success_rate={final['success_rate']:.3f} cost_rate={final['cost_rate']:.3f} "
          f"(target {d_threshold})")

    # Convergence check on lambda's last third of updates: does it stay
    # within a reasonably narrow band, or is it still swinging widely?
    last_third = history[-max(1, len(history) // 3):]
    lambdas = [r["lambda"] for r in last_third]
    lam_range = max(lambdas) - min(lambdas)
    if lam_range > 0.5 * (max(lambdas) if max(lambdas) > 0 else 1.0):
        print(f"\n  WARNING: lambda is still swinging widely in the final third of "
              f"training (range={lam_range:.3f} over values {lambdas}). This suggests "
              f"eta may be too large (overcorrecting each update) relative to how fast "
              f"the policy can actually respond -- consider a smaller eta candidate, or "
              f"re-running tune_lagrangian_eta_oversight.py with UPDATE_INTERVAL raised "
              f"(fewer, larger updates may stabilize this) before trusting this "
              f"architecture's results in the real sweep.")
    else:
        print(f"\n  Lambda looks reasonably converged in the final third of training "
              f"(range={lam_range:.3f}) -- looks safe to proceed to the real sweep.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    steps = [r["steps"] for r in history]
    axes[0].plot(steps, [r["success_rate"] for r in history], marker="o")
    axes[0].set_xlabel("training steps"); axes[0].set_ylabel("success_rate")
    axes[0].set_title("Does success_rate climb?")

    axes[1].plot(steps, [r["cost_rate"] for r in history], marker="o", color="tab:red")
    axes[1].axhline(d_threshold, color="gray", linestyle="--", label=f"target ({d_threshold})")
    axes[1].set_xlabel("training steps"); axes[1].set_ylabel("cost_rate (violation_rate)")
    axes[1].set_title("Does cost_rate settle near the target?")
    axes[1].legend()

    axes[2].plot(steps, [r["lambda"] for r in history], marker="o", color="tab:purple")
    axes[2].set_xlabel("training steps"); axes[2].set_ylabel("lambda")
    axes[2].set_title("Does lambda converge (not oscillate/diverge)?")

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sanity_check_lagrangian_oversight_learning_curves.png", dpi=150)
    print("\nSaved results/sanity_check_lagrangian_oversight_learning_curves.png")
