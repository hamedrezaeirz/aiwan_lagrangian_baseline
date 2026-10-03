"""
merge_results_lagrangian_oversight.py
-----------------------------------------
Merges every results/results_lagrangian_oversight_seeds_*.csv produced
by parallel `--seeds` runs into results/results_lagrangian_oversight.csv.
Mirrors merge_results_oversight.py.

Usage (after every --seeds process has finished):
    python3 merge_results_lagrangian_oversight.py

Does NOT deduplicate or verify completeness -- ALWAYS run
    python3 check_completeness_oversight.py results/results_lagrangian_oversight.csv
afterward (expects 1,080 rows: 20 seeds x 9 cells x 6 conditions).
"""

import glob
import pandas as pd

from config_oversight import RESULTS_DIR

OUT_PATH = f"{RESULTS_DIR}/results_lagrangian_oversight.csv"

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RESULTS_DIR}/results_lagrangian_oversight_seeds_*.csv"))
    if not files:
        raise SystemExit(f"No results_lagrangian_oversight_seeds_*.csv files found in "
                         f"{RESULTS_DIR}/ -- nothing to merge yet.")
    print(f"Merging {len(files)} files:")
    for f in files:
        print(f"  {f}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(df)} total rows to {OUT_PATH}")
    print(f"\nNow run: python3 check_completeness_oversight.py {OUT_PATH}")
