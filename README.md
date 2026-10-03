# aiwan_lagrangian_baseline

Code, data, and figures for **Experiment 3** of the AIWAN paper (*AIWAN: Artificial Intelligence With Artificial Need*):
a stronger single-loop baseline, **Architecture D = Lagrangian PPO**, compared with Architectures A, B, and C
(§7.8 of the paper).

## What D is
D maximizes `r_goal - lambda_t * c_t` plus the environment's own dense, non-terminal shaping, where `c_t = 1` on the step
a constraint is violated (battery depleted / oversight violation). The multiplier replaces the environment's fixed -50
terminal penalty and is updated by dual ascent, `lambda <- max(0, lambda + eta * (cost_rate - 0.05))`, with at most 20
updates per run. Hyperparameters (`results/hyperparams_lagrangian_*.json`) were tuned on validation seeds 100-102,
disjoint from the sweep's seeds, at 200,000 steps (battery: under override w = 8; oversight: standard condition).

## Status
- **Battery signal: complete.** 540 rows (10 seeds x 9 cells x 6 conditions) in `results/results_lagrangian_battery.csv`; analysis in `results/analysis_output_battery_D.txt`.
- **Oversight signal: complete.** 1,080 rows (20 seeds x 9 cells x 6 conditions) in `results/results_lagrangian_oversight.csv`; analysis in `results/analysis_output_oversight_D.txt`.
  The oversight sweep was started with `launch_parallel_lagrangian_oversight.sh` (2 processes) and, partway through, split across 4 processes by stopping each process after a completed cell and relaunching the remaining seeds (the runner is resume-safe: completed cells are skipped). The merged file was checked with `check_completeness_oversight.py` (1,080 unique keys, no duplicates).

## Files
- `lagrangian_ppo.py`: shared Lagrangian-PPO core. `train_architecture_d_*.py`: per-signal wrappers.
- `run_experiment_lagrangian_*.py`, `launch_parallel_lagrangian_*.sh`, `merge_results_lagrangian_*.py`, `check_completeness_*.py`: sweeps and bookkeeping.
- `tune_lagrangian_eta_*.py`, `sanity_check_lagrangian_*.py`, `check_500k.py`: tuning and sanity checks.
- `plot_results_lagrangian_battery.py`, `plot_results_lagrangian_oversight.py`: analysis (per-cell paired tests, nothing pooled).
- `env_*.py`, `config_*.py`, `evaluate_*.py`: copies of the environments, configuration, and evaluation used by the A/B/C experiments, so this repository is self-contained.
- `results/results_battery.csv`, `results/results_oversight.csv`: the A/B/C results (also in the `aiwan_battery_continuous` and `aiwan_oversight` repositories), needed by the plot scripts.

## Reproduce
```
pip install stable-baselines3 gymnasium torch pandas scipy matplotlib numpy
# battery
python3 tune_lagrangian_eta_battery.py
bash launch_parallel_lagrangian_battery.sh
python3 merge_results_lagrangian_battery.py
python3 check_completeness_battery.py results/results_lagrangian_battery.csv
python3 plot_results_lagrangian_battery.py results/results_battery.csv results/results_lagrangian_battery.csv
# oversight
python3 tune_lagrangian_eta_oversight.py
bash launch_parallel_lagrangian_oversight.sh
python3 merge_results_lagrangian_oversight.py
python3 check_completeness_oversight.py results/results_lagrangian_oversight.csv
python3 plot_results_lagrangian_oversight.py results/results_oversight.csv results/results_lagrangian_oversight.csv
```

## Notes on interpretation
- **Battery.** eta = 0.01 is the smallest candidate in the tuning grid and lambda can reach at most about 0.2 (20 updates x 0.01), so D behaves like an agent trained on the goal reward plus dense shaping with a negligible crash penalty; in a single diagnostic run (`check_500k.py`) lambda rose to about 0.006 and then fell to zero. The constraint is not binding in that environment. At 100k and 500k steps D matches C's crash rate with higher success; at 10k steps D fails like A.
- **Oversight.** eta = 0.5; lambda can reach at most about 10, against A's fixed terminal penalty of 25 (0.5 x -50). D improves on A (violation rate lower by 0.06-0.26 at 100k and 500k steps) but stays far from C: at 500k steps D's violation rate is 0.385-0.419, against at most 0.002 for C, and it does not reach its own target of 0.05. Lambda was not logged in the sweeps, and eta was not re-tuned after seeing results.
- Always read violation/crash rate together with success rate; nothing is pooled across cells.
