"""
plot_results_lagrangian_battery.py  (v2: success panels + per-cell tests)
-------------------------------------------------------------------------
Compares Architecture D (Lagrangian-PPO, Experiment 3) with A, B, C on the
battery signal in the continuous environment. All tests are paired by seed
and computed PER CELL (nothing pooled). crash_rate is always shown with
success_rate (an inert policy never crashes).

Produces:
  results/battery_D_vs_budget_w8.png  : crash (left) and success (right) vs
                                        training budget at override_w8,
                                        largest network size (A, B, C, D)
  results/battery_D_vs_w_largest.png  : crash and success vs override weight w,
                                        largest cell
  + printed per-cell table, per-cell paired tests (D vs A, D vs C), and D's
    `standard` condition.

Usage:
    python3 plot_results_lagrangian_battery.py \
        [results/results_battery.csv] [results/results_lagrangian_battery.csv]
"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY = ["architecture", "network_size", "training_timesteps", "seed", "condition"]
NETS = [16, 256, 2048]
BUDGETS = [10_000, 100_000, 500_000]
W = 8
STYLES = {
    "A": ("--", "#d64545", "A (reward-shaping)"),
    "B": ("-.", "#e0a030", "B (hardcoded sigmoid gate)"),
    "C": ("-", "#3ba55d", "C (learned gate)"),
    "D": (":", "#8855dd", "D (Lagrangian-PPO)"),
}


def load(abc_path, d_path):
    abc, d = pd.read_csv(abc_path), pd.read_csv(d_path)
    for name, df in (("A/B/C", abc), ("D", d)):
        if (df.groupby(KEY).size() > 1).any():
            raise SystemExit(f"Duplicate keys in {name} file -- run the completeness check first.")
        df["net"] = df["network_size"].astype(str).str.strip("[]").astype(int)
    print(f"Loaded {len(abc)} A/B/C rows ({abc.seed.nunique()} seeds), {len(d)} D rows ({d.seed.nunique()} seeds).")
    if len(d) != 540:
        print(f"WARNING: D has {len(d)} rows, expected 540.")
    return abc, d


def series(abc, d, arch, net, bud, cond, metric):
    df = d if arch == "D" else abc
    q = df[(df.architecture == arch) & (df.net == net) & (df.training_timesteps == bud) & (df.condition == cond)]
    return q.set_index("seed")[metric].sort_index()


def sem(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def per_cell_table(abc, d, cond):
    print(f"\n=== per-cell means, condition = {cond}: crash A/B/C/D | success A/B/C/D ===")
    for net in NETS:
        for b in BUDGETS:
            cr = [series(abc, d, a, net, b, cond, "crash_rate").mean() for a in "ABCD" if not (a == "D" and cond not in set(d.condition))]
            su = [series(abc, d, a, net, b, cond, "success_rate").mean() for a in "ABCD" if not (a == "D" and cond not in set(d.condition))]
            print(f"  net={net:<5} budget={b:<7} crash {' '.join(f'{v:.3f}' for v in cr)} | success {' '.join(f'{v:.3f}' for v in su)}")


def test(x, y):
    diff = (x - y)
    if np.std(diff, ddof=1) == 0:
        return diff.mean(), "t undefined"
    t, p = stats.ttest_1samp(diff, 0)
    return diff.mean(), f"t(9)={t:.2f}, p={p:.3g}"


def cell_tests(abc, d):
    print(f"\n=== per-cell paired tests at override_w{W}: D minus reference (positive = D higher) ===")
    for net in NETS:
        for b in BUDGETS:
            for ref in ("A", "C"):
                for m in ("crash_rate", "success_rate"):
                    dm, ts = test(series(abc, d, "D", net, b, f"override_w{W}", m),
                                  series(abc, d, ref, net, b, f"override_w{W}", m))
                    print(f"  net={net:<5} budget={b:<7} D-{ref} {m[:7]:<7} {dm:+.3f}  {ts}")


def figures(abc, d):
    net = NETS[-1]
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, m in zip(axes, ("crash_rate", "success_rate")):
        for arch, (ls, c, lab) in STYLES.items():
            ms = [series(abc, d, arch, net, b, f"override_w{W}", m).mean() for b in BUDGETS]
            es = [sem(series(abc, d, arch, net, b, f"override_w{W}", m)) for b in BUDGETS]
            ax.errorbar(BUDGETS, ms, yerr=es, fmt=ls, marker="o", color=c, label=lab, capsize=2)
        ax.set_xscale("log"); ax.set_xlabel("training budget (steps)"); ax.set_ylabel(m)
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"override_w{W}, network size {net} (mean +- SEM, 10 seeds)")
    fig.tight_layout(); fig.savefig("results/battery_D_vs_budget_w8.png", dpi=150); plt.close(fig)
    print("Saved results/battery_D_vs_budget_w8.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, m in zip(axes, ("crash_rate", "success_rate")):
        for arch, (ls, c, lab) in STYLES.items():
            ws = (0, 1, 2, 4, 8)
            ms = [series(abc, d, arch, net, BUDGETS[-1], f"override_w{w}", m).mean() for w in ws]
            es = [sem(series(abc, d, arch, net, BUDGETS[-1], f"override_w{w}", m)) for w in ws]
            ax.errorbar(ws, ms, yerr=es, fmt=ls, marker="o", color=c, label=lab, capsize=2)
        ax.set_xlabel("override / conflict weight (w)"); ax.set_ylabel(m)
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"network size {net}, budget {BUDGETS[-1]:,} (mean +- SEM, 10 seeds)")
    fig.tight_layout(); fig.savefig("results/battery_D_vs_w_largest.png", dpi=150); plt.close(fig)
    print("Saved results/battery_D_vs_w_largest.png")


if __name__ == "__main__":
    abc_path = sys.argv[1] if len(sys.argv) > 1 else "results/results_battery.csv"
    d_path = sys.argv[2] if len(sys.argv) > 2 else "results/results_lagrangian_battery.csv"
    abc, d = load(abc_path, d_path)
    per_cell_table(abc, d, f"override_w{W}")
    cell_tests(abc, d)
    print("\n=== Architecture D, `standard` condition per cell: crash | success ===")
    for net in NETS:
        print(f"  net={net:<5}", [(round(series(abc, d, 'D', net, b, 'standard', 'crash_rate').mean(), 3),
                                    round(series(abc, d, 'D', net, b, 'standard', 'success_rate').mean(), 3)) for b in BUDGETS])
    figures(abc, d)
    print("\nRead crash_rate together with success_rate; nothing above is pooled across cells.")
