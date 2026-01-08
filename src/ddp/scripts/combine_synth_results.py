"""Combine synthetic result CSVs into a single per-trial table."""

from __future__ import annotations

import argparse
from pathlib import Path
import glob

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine synthetic result CSVs into a single per-trial file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing CSV files to combine",
    )
    parser.add_argument(
        "--pattern",
        default="seed*_d*.csv",
        help="Glob pattern to match CSV files (default: seed*_d*.csv)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["_jobs.csv", "opt_only"],
        help="Substrings to exclude from filenames (default: _jobs.csv, opt_only)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    # Find all matching CSV files
    input_dir = Path(args.input_dir)
    pattern = args.pattern
    exclude = args.exclude
    
    csv_files = sorted(input_dir.glob(pattern))
    
    # Filter out excluded files
    filtered_files = []
    for f in csv_files:
        if not any(exc in str(f) for exc in exclude):
            filtered_files.append(f)
    
    print(f"Found {len(filtered_files)} CSV files to combine")
    
    if not filtered_files:
        raise ValueError(f"No CSV files found matching pattern '{pattern}' in {input_dir}")
    
    # Load and combine
    dfs = []
    for f in filtered_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {f}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid CSV files were loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by d, seed, shadow, dispatch if available
    sort_cols = ['d', 'seed', 'shadow', 'dispatch']
    available_sort_cols = [col for col in sort_cols if col in combined.columns]
    if available_sort_cols:
        combined = combined.sort_values(available_sort_cols, na_position='last')
    
    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    
    print(f"Combined {len(combined)} rows into {out_path}")
    print(f"Columns: {', '.join(combined.columns.tolist())}")


if __name__ == "__main__":
    main()

