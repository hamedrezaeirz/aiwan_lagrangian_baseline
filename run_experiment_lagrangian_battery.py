"""
run_experiment_lagrangian_battery.py
----------------------------------------
Runs Architecture D (Lagrangian-PPO, Experiment 3) on the battery
signal across the SAME NETWORK_SIZES x TRAINING_BUDGETS x SEEDS grid as
the finished A/B/C battery sweep (config_battery.py). Mirrors
run_experiment_lagrangian_oversight.py exactly, with:
  - cost_key="crashed", signal_key="r_battery" (via
    train_architecture_d_battery.train_architecture_d)
  - config_battery.SEEDS (10 seeds, 0-9), not 20 -- battery is a
    robustness check, not the primary result, same split as A/B/C's
    own battery sweep
  - evaluate_battery.py's row schema (crash_rate, avg_battery_level in
    place of violation_rate, avg_activations_per_episode)

Conditions run for D (same design decision as oversight): `standard`
plus the override sweep (override_w0..w8) = 6 rows per
(net_size, budget, seed) cell -- goal_nulling/indifference intentionally
skipped (see run_experiment_lagrangian_oversight.py's docstring for the
full reasoning; identical here).

Expected output: 10 seeds x 9 cells x 6 conditions = 540 rows.

Hyperparameters come from results/hyperparams_lagrangian_battery.json
(tune_lagrangian_eta_battery.py's output).

Usage:
    python run_experiment_lagrangian_battery.py             # full sweep
    python run_experiment_lagrangian_battery.py --quick     # pipeline check (minutes)
    python run_experiment_lagrangian_battery.py --mini      # one real-budget cell, 2 seeds
    python run_experiment_lagrangian_battery.py --seeds 0 1 2 3 4
                                                              # subset (parallelize)

Resume: identical crash-safe behavior to
run_experiment_lagrangian_oversight.py -- see that file's docstring.
"""

import argparse
import csv
import json
import os

import torch

torch.set_num_threads(1)

from config_battery import (
    NETWORK_SIZES, TRAINING_BUDGETS, SEEDS, OVERRIDE_WEIGHTS,
    RESULTS_DIR, N_EVAL_EPISODES,
)
from train_architecture_d_battery import train_architecture_d
from evaluate_battery import evaluate_cell, as_policy_fn

HYPERPARAMS_PATH_D = f"{RESULTS_DIR}/hyperparams_lagrangian_battery.json"
RESULTS_CSV_PATH_D = f"{RESULTS_DIR}/results_lagrangian_battery.csv"
N_DUAL_UPDATES = 20

CONDITIONS = [("standard", 0.0)] + [(f"override_w{w}", float(w)) for w in OVERRIDE_WEIGHTS]

ROW_FIELDS = [
    "architecture", "network_size", "training_timesteps", "seed", "condition",
    "n_eval_episodes", "success_rate", "crash_rate",
    "avg_battery_level", "avg_episode_length",
]


def load_hp():
    if not os.path.exists(HYPERPARAMS_PATH_D):
        raise SystemExit(
            f"{HYPERPARAMS_PATH_D} not found. Run tune_lagrangian_eta_battery.py "
            f"first (or copy its output over) -- the real sweep must not run on "
            f"un-tuned default hyperparameters."
        )
    with open(HYPERPARAMS_PATH_D) as f:
        return json.load(f)


def _clear_stale_test_output(path):
    if os.path.exists(path):
        print(f"NOTE: removing stale {path} from a previous run before starting.")
        os.remove(path)


def _cell_key(row):
    return (row["network_size"], int(row["training_timesteps"]), int(row["seed"]))


def prepare_resume(path, expected_rows_per_cell):
    """Identical logic to run_experiment_lagrangian_oversight.py's
    prepare_resume -- see that file for the full docstring."""
    if not os.path.exists(path):
        return set()

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_cell = {}
    for r in rows:
        by_cell.setdefault(_cell_key(r), []).append(r)

    complete = {k for k, v in by_cell.items() if len(v) == expected_rows_per_cell}
    partial = {k for k, v in by_cell.items() if len(v) != expected_rows_per_cell}

    if partial:
        print(f"NOTE: found {len(partial)} partially-written cell(s) in {path} "
              f"({sorted(partial)}); discarding their rows so they are redone "
              f"cleanly (no duplicates).")
        kept = [r for r in rows if _cell_key(r) in complete]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
            writer.writeheader()
            writer.writerows(kept)

    if complete:
        print(f"Resume: {len(complete)} cell(s) already complete in {path} -- skipping them.")
    return complete


def write_rows(rows, path):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run(quick=False, mini=False, seeds=None):
    hp = load_hp()
    eta = hp.get("lagrangian_eta")
    d_threshold = hp.get("lagrangian_d_threshold", 0.05)
    if eta is None:
        raise SystemExit(f"{HYPERPARAMS_PATH_D} has no 'lagrangian_eta' -- was "
                         f"tune_lagrangian_eta_battery.py's eta phase completed?")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if quick:
        network_sizes, training_budgets, run_seeds = [[16]], [2_000], [0]
        out_path = f"{RESULTS_DIR}/results_lagrangian_battery_quick_test.csv"
        _clear_stale_test_output(out_path)
        print(f"Running --quick: tiny grid, just validating the pipeline. Writing to {out_path}.")
    elif mini:
        network_sizes, training_budgets, run_seeds = [[64]], [100_000], [0, 1]
        out_path = f"{RESULTS_DIR}/results_lagrangian_battery_mini_test.csv"
        _clear_stale_test_output(out_path)
        print(f"Running --mini: ONE real-budget cell (net=[64], budget=100,000), "
              f"seeds 0-1, all {len(CONDITIONS)} conditions. Pipeline validation, "
              f"not conclusions. Writing to {out_path}.")
    else:
        network_sizes, training_budgets = NETWORK_SIZES, TRAINING_BUDGETS
        run_seeds = seeds if seeds is not None else SEEDS
        if seeds is not None:
            tag = "_".join(str(s) for s in run_seeds)
            out_path = f"{RESULTS_DIR}/results_lagrangian_battery_seeds_{tag}.csv"
        else:
            out_path = RESULTS_CSV_PATH_D
        print(f"Running seeds {run_seeds} (of {SEEDS}). Writing to {out_path}.")

    print(f"Hyperparameters: {hp}\n")

    completed = prepare_resume(out_path, expected_rows_per_cell=len(CONDITIONS)) \
        if not (quick or mini) else set()

    for seed in run_seeds:
        for net_size in network_sizes:
            for budget in training_budgets:
                key = (str(net_size), budget, seed)
                if key in completed:
                    print(f"--- seed={seed} net_size={net_size} budget={budget} --- (already complete, skipping)")
                    continue

                print(f"--- seed={seed} net_size={net_size} budget={budget} ---")
                rows = []
                for cond_name, w in CONDITIONS:
                    model_d = train_architecture_d(
                        net_size, budget, seed, hp,
                        adversarial_weight=w, d_threshold=d_threshold,
                        eta=eta, n_updates=N_DUAL_UPDATES,
                    )
                    rows.append(evaluate_cell(
                        "D", net_size, budget, seed, cond_name,
                        as_policy_fn(model_d), N_EVAL_EPISODES,
                        adversarial_weight=w,
                    ))
                write_rows(rows, out_path)

    print(f"Done. Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.quick and args.mini:
        raise SystemExit("--quick and --mini are mutually exclusive -- pick one.")
    run(quick=args.quick, mini=args.mini, seeds=args.seeds)

    if args.seeds is not None:
        print("\nWhen every --seeds process is done, merge with "
              "merge_results_lagrangian_battery.py.")
