#!/bin/bash
# launch_parallel_lagrangian_battery.sh
# -----------------------------------------
# Splits config_battery.SEEDS = [0..9] across 2 processes (5 seeds
# each) -- same 2-cores-only convention as this project's other
# launch_parallel_*.sh scripts.
#
#   nohup bash launch_parallel_lagrangian_battery.sh > launch_lagrangian_battery_main.log 2>&1 &
#   disown
#
# Progress:
#   tail -n 5 logs/lagrangian_battery_seeds_0-4.log
#   tail -n 5 logs/lagrangian_battery_seeds_5-9.log
#   wc -l results/results_lagrangian_battery_seeds_*.csv   # target: 271 each (1 header + 5x9x6)

set -e
cd "$(dirname "$0")"
if [ -z "$VIRTUAL_ENV" ] && [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi
mkdir -p results logs

echo "Starting 2 parallel processes across seeds 0-9 (5 seeds each)..."

python3 -u run_experiment_lagrangian_battery.py --seeds 0 1 2 3 4 \
    > logs/lagrangian_battery_seeds_0-4.log 2>&1 &
PID1=$!
python3 -u run_experiment_lagrangian_battery.py --seeds 5 6 7 8 9 \
    > logs/lagrangian_battery_seeds_5-9.log 2>&1 &
PID2=$!

echo "Launched PIDs: $PID1 $PID2"
wait $PID1 $PID2

echo "Both processes finished. Run: python3 merge_results_lagrangian_battery.py"
