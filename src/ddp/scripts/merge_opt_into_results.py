"""Merge OPT values from OPT-only results into main result CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge OPT values from OPT-only results into main results CSV."
    )
    parser.add_argument(
        "--main-results",
        type=Path,
        required=True,
        help="Path to main results CSV (all policies, may or may not have OPT)",
    )
    parser.add_argument(
        "--opt-results",
        type=Path,
        required=True,
        help="Path to OPT-only results CSV (contains opt+opt rows with savings=opt_total)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    # Load main results
    print("Loading main results...")
    main_df = pd.read_csv(args.main_results)
    print(f"Loaded {len(main_df)} rows from main results")
    print(f"Columns: {', '.join(main_df.columns.tolist())}")

    # Add param and param_value columns for sweep mode compatibility
    print("Adding parameter columns for sweep mode...")
    main_df['param'] = 'd'
    main_df['param_value'] = pd.to_numeric(main_df['d'], errors='coerce')

    # Load OPT results and extract lookup
    print("\nLoading OPT results...")
    opt_df = pd.read_csv(args.opt_results)
    print(f"Loaded {len(opt_df)} rows from OPT results")
    
    # Extract non-opt policies from opt_only that are missing from main
    print("Checking for missing policies in main results...")
    opt_only_policies = opt_df[
        ~((opt_df['shadow'] == 'opt') & (opt_df['dispatch'] == 'opt'))
    ].copy()
    
    if not opt_only_policies.empty:
        # Find policies in opt_only that aren't in main
        main_policy_keys = set(
            main_df.apply(lambda row: f"{row['shadow']}+{row['dispatch']}", axis=1)
        )
        opt_only_policy_keys = set(
            opt_only_policies.apply(lambda row: f"{row['shadow']}+{row['dispatch']}", axis=1)
        )
        missing_policies = opt_only_policy_keys - main_policy_keys
        
        if missing_policies:
            print(f"Found missing policies in opt_only results: {missing_policies}")
            missing_rows = opt_only_policies[
                opt_only_policies.apply(
                    lambda row: f"{row['shadow']}+{row['dispatch']}" in missing_policies,
                    axis=1
                )
            ].copy()
            # Add param and param_value columns to missing rows for sweep mode compatibility
            missing_rows['param'] = 'd'
            missing_rows['param_value'] = pd.to_numeric(missing_rows['d'], errors='coerce')
            print(f"Adding {len(missing_rows)} rows from opt_only to main results")
            main_df = pd.concat([main_df, missing_rows], ignore_index=True)
        else:
            print("No missing policies found")
    
    # Extract rows where shadow="opt" and dispatch="opt"
    # The savings column contains the opt_total value
    opt_rows = opt_df[
        (opt_df['shadow'] == 'opt') & 
        (opt_df['dispatch'] == 'opt')
    ].copy()
    
    if opt_rows.empty:
        raise ValueError("No opt+opt rows found in OPT results CSV. Expected rows with shadow='opt' and dispatch='opt'.")
    
    print(f"Found {len(opt_rows)} opt+opt rows")
    
    # Create lookup table: (seed, d, n) -> opt_total (from savings column)
    opt_lookup = opt_rows[['seed', 'd', 'n', 'savings']].copy()
    opt_lookup = opt_lookup.rename(columns={'savings': 'opt_total'})
    
    # Convert to numeric and ensure consistent types
    opt_lookup['seed'] = pd.to_numeric(opt_lookup['seed'], errors='coerce')
    opt_lookup['d'] = pd.to_numeric(opt_lookup['d'], errors='coerce')
    opt_lookup['n'] = pd.to_numeric(opt_lookup['n'], errors='coerce')
    opt_lookup['opt_total'] = pd.to_numeric(opt_lookup['opt_total'], errors='coerce')
    
    opt_lookup = opt_lookup.dropna(subset=['opt_total', 'seed', 'd', 'n'])
    opt_lookup = opt_lookup.drop_duplicates(subset=['seed', 'd', 'n'])
    print(f"Created lookup with {len(opt_lookup)} unique (seed, d, n) combinations with OPT")

    # Merge OPT values into main results
    print("\nMerging OPT values...")
    
    # Ensure main_df has consistent numeric types for merge keys
    main_df['seed'] = pd.to_numeric(main_df['seed'], errors='coerce')
    main_df['d'] = pd.to_numeric(main_df['d'], errors='coerce')
    main_df['n'] = pd.to_numeric(main_df['n'], errors='coerce')
    main_df['savings'] = pd.to_numeric(main_df['savings'], errors='coerce')
    
    # Merge OPT values
    main_df = main_df.merge(
        opt_lookup[['seed', 'd', 'n', 'opt_total']],
        on=['seed', 'd', 'n'],
        how='left',
        suffixes=('', '_new')
    )

    # If opt_total column already exists, combine values
    if 'opt_total' in main_df.columns and 'opt_total_new' in main_df.columns:
        main_df['opt_total'] = main_df['opt_total'].fillna(main_df['opt_total_new'])
        main_df = main_df.drop(columns=['opt_total_new'])
    elif 'opt_total_new' in main_df.columns:
        main_df['opt_total'] = main_df['opt_total_new']
        main_df = main_df.drop(columns=['opt_total_new'])

    # Compute opt_gap and ratio_opt for all rows
    print("Computing opt_gap and ratio_opt...")
    if 'opt_total' in main_df.columns and 'savings' in main_df.columns:
        # Compute opt_gap = max(opt_total - savings, 0)
        main_df['opt_gap'] = (main_df['opt_total'] - main_df['savings']).clip(lower=0)
        
        # Compute ratio_opt = savings / opt_total (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            main_df['ratio_opt'] = main_df['savings'] / main_df['opt_total'].replace(0, np.nan)
        
        # Set ratio_opt to NaN where opt_total is missing or zero
        main_df['ratio_opt'] = main_df['ratio_opt'].where(main_df['opt_total'].notna() & (main_df['opt_total'] != 0))
    else:
        print("Warning: Missing 'opt_total' or 'savings' columns, skipping metric computation")

    # Sort similar to Meituan format (by d, seed, shadow, dispatch)
    sort_cols = ['d', 'seed', 'shadow', 'dispatch']
    available_sort_cols = [col for col in sort_cols if col in main_df.columns]
    if available_sort_cols:
        main_df = main_df.sort_values(available_sort_cols, na_position='last')

    # Save merged file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(output_path, index=False)
    
    print(f"\nSaved merged file to {output_path}")
    print(f"Total rows: {len(main_df)}")
    print(f"Rows with OPT: {main_df['opt_total'].notna().sum()}")
    print(f"Rows with ratio_opt: {main_df['ratio_opt'].notna().sum()}")
    print(f"Rows with opt_gap: {main_df['opt_gap'].notna().sum()}")
    print(f"\nColumns: {', '.join(main_df.columns.tolist())}")


if __name__ == "__main__":
    main()

