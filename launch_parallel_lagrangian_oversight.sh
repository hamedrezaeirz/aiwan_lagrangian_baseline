#!/bin/bash
# launch_parallel_lagrangian_oversight.sh
# -----------------------------------------
# Splits SEEDS 0-19 across 2 processes (10 seeds each) -- the same
# 2-cores-only convention as launch_parallel_oversight.sh /
# launch_parallel_battery.sh (run_experiment_lagrangian_oversight.py sets
# torch.set_num_threads(1), so 2 processes = exactly 2 cores).
#
# Each process writes its own results/results_lagrangian_oversight_seeds_<...>.csv.
# Merge afterwards with merge_results_lagrangian_oversight.py, then verify
# with check_completeness_oversight.py (expects 1,080 rows).
#
# Safe to re-run after an interruption: finished cells are skipped, a
# partially-written cell is redone (see run_experiment_lagrangian_oversight.py).
#
#   nohup bash launch_parallel_lagrangian_oversight.sh > launch_lagrangian_main.log 2>&1 &
#   disown
#
# Progress:
#   tail -n 5 logs/lagrangian_seeds_0-9.log
#   tail -n 5 logs/lagrangian_seeds_10-19.log
#   wc -l results/results_lagrangian_oversight_seeds_*.csv   # target: 541 each (1 header + 10x9x6)

set -e
cd "$(dirname "$0")"
if [ -z "$VIRTUAL_ENV" ] && [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi
mkdir -p results logs

echo "Starting 2 parallel processes across seeds 0-19 (10 seeds each)..."

python3 -u run_experiment_lagrangian_oversight.py --seeds 0 1 2 3 4 5 6 7 8 9 \
    > logs/lagrangian_seeds_0-9.log 2>&1 &
PID1=$!
python3 -u run_experiment_lagrangian_oversight.py --seeds 10 11 12 13 14 15 16 17 18 19 \
    > logs/lagrangian_seeds_10-19.log 2>&1 &
PID2=$!

echo "Launched PIDs: $PID1 $PID2"
wait $PID1 $PID2

echo "Both processes finished. Run: python3 merge_results_lagrangian_oversight.py"
