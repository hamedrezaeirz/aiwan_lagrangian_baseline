"""
merge_results_lagrangian_battery.py
---------------------------------------
Merges every results/results_lagrangian_battery_seeds_*.csv into
results/results_lagrangian_battery.csv. Mirrors
merge_results_lagrangian_oversight.py.

Usage (after every --seeds process has finished):
    python3 merge_results_lagrangian_battery.py

Does NOT deduplicate or verify completeness -- ALWAYS run
    python3 check_completeness_battery.py results/results_lagrangian_battery.csv
afterward (expects 540 rows: 10 seeds x 9 cells x 6 conditions).
"""

import glob
import pandas as pd

from config_battery import RESULTS_DIR

OUT_PATH = f"{RESULTS_DIR}/results_lagrangian_battery.csv"

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RESULTS_DIR}/results_lagrangian_battery_seeds_*.csv"))
    if not files:
        raise SystemExit(f"No results_lagrangian_battery_seeds_*.csv files found in "
                         f"{RESULTS_DIR}/ -- nothing to merge yet.")
    print(f"Merging {len(files)} files:")
    for f in files:
        print(f"  {f}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} total rows to {OUT_PATH}")
    print(f"\nNow run: python3 check_completeness_battery.py {OUT_PATH}")
