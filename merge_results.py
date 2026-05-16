"""
merge_results.py
----------------
Merges summary CSVs produced by the four per-GPU scripts into a single
combined summary. Run after all GPU scripts have finished.

Usage
-----
    python3 merge_results.py

Outputs
-------
results/combined_summary.csv   — merged table, one row per (method, n_assets, seed)
results/combined_summary.txt   — console-style pivot printed to file
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


GPU_DIRS = [
    Path("results/gpu0"),   # n_assets=1
    Path("results/gpu1"),   # n_assets=5
    Path("results/gpu2"),   # n_assets=10
    Path("results/gpu3"),   # n_assets=20
    Path("results/gpu4"),   # n_assets=100
]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)


def load_gpu_results(gpu_dir: Path) -> pd.DataFrame | None:
    csv = gpu_dir / "summary.csv"
    if not csv.exists():
        print(f"  WARNING: {csv} not found — skipping (GPU may not have finished)")
        return None
    df = pd.read_csv(csv)
    print(f"  Loaded {len(df):4d} rows from {csv}")
    return df


def main():
    print("=" * 60)
    print("  Merging GPU results")
    print("=" * 60)

    frames = []
    for d in GPU_DIRS:
        df = load_gpu_results(d)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No results found. Make sure at least one GPU script has finished.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # Drop exact duplicate rows (can happen if --resume replayed a checkpoint)
    key_cols = ["method", "n_assets", "seed", "goal_mult"]
    key_cols = [c for c in key_cols if c in combined.columns]
    combined = combined.drop_duplicates(subset=key_cols, keep="last")

    # Sort for readability
    combined = combined.sort_values(
        key_cols, ascending=True
    ).reset_index(drop=True)

    # Save
    out_csv = OUT_DIR / "combined_summary.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\nCombined summary → {out_csv}  ({len(combined)} rows)")

    # Print goal-probability pivot
    mc_col = "mc_goal_prob" if "mc_goal_prob" in combined.columns else "goal_probability"
    if mc_col in combined.columns:
        pivot = (
            combined.groupby(["method", "n_assets"])[mc_col]
            .mean()
            .unstack("n_assets")
        )
        # Sort by first available n_assets column (descending goal prob)
        first_col = pivot.columns[0]
        pivot = pivot.sort_values(first_col, ascending=False)

        print("\n" + "=" * 60)
        print("  MC GOAL PROBABILITY  (mean across seeds)")
        print("=" * 60)
        with pd.option_context("display.float_format", "{:.1%}".format,
                               "display.max_rows", 60,
                               "display.max_columns", 20):
            txt = pivot.to_string()
            print(txt)

        txt_path = OUT_DIR / "combined_summary.txt"
        txt_path.write_text(txt + "\n")
        print(f"\nPivot table → {txt_path}")

    # Print training-time summary
    if "train_time_sec" in combined.columns:
        print("\n" + "=" * 60)
        print("  MEAN TRAINING TIME (seconds, across n_assets & seeds)")
        print("=" * 60)
        time_df = (
            combined.groupby("method")[["train_time_sec"]]
            .mean()
            .sort_values("train_time_sec", ascending=False)
        )
        with pd.option_context("display.float_format", "{:.1f}".format):
            print(time_df.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
