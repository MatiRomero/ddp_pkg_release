"""Aggregate per-trial SHADOW×DISPATCH CSV results into summary statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

# Columns that identify a unique experimental configuration.
GROUP_FIELDS: list[str] = ["param", "param_value", "shadow", "dispatch", "n", "d"]

# Numeric metrics recorded for each trial that we aggregate with mean/std.
NUMERIC_FIELDS: list[str] = [
    "savings",
    "pooled_pct",
    "ratio_lp",
    "ratio_opt",
    "pairs",
    "solos",
    "time_s",
]


def _clean_category(series: pd.Series, *, lower: bool = False) -> pd.Series:
    """Strip whitespace, optionally lowercase, and preserve missing values."""

    def _clean(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return pd.NA
            return text.lower() if lower else text
        return value

    return series.map(_clean)


def _ensure_parent_dir(path: Path) -> None:
    parent = path.expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def aggregate(path: str) -> pd.DataFrame:
    """Aggregate a per-trial CSV file into mean/std metrics by configuration.

    Parameters
    ----------
    path:
        Path to the CSV produced by :mod:`ddp.scripts.sweep_param` or a similar
        runner that emits per-trial rows.

    Returns
    -------
    pandas.DataFrame
        Aggregated data where each row corresponds to a unique combination of
        ``(param, param_value, shadow, dispatch, n, d)`` with mean and std
        columns (``mean_*`` and ``std_*``) for each metric in
        :data:`NUMERIC_FIELDS` that is present in the input CSV.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure group-by columns exist even if the CSV omitted them.
    for col in GROUP_FIELDS:
        if col not in df.columns:
            df[col] = pd.NA

    if "shadow" in df.columns:
        df["shadow"] = _clean_category(df["shadow"], lower=True)
    if "dispatch" in df.columns:
        df["dispatch"] = _clean_category(df["dispatch"], lower=True)
    if "param" in df.columns:
        df["param"] = _clean_category(df["param"], lower=True)

    # Convert numeric-like columns to floats (non-numeric entries become NaN).
    numeric_candidates: Iterable[str] = list(NUMERIC_FIELDS) + ["n", "d", "param_value"]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    numeric_cols = [col for col in NUMERIC_FIELDS if col in df.columns]
    if not numeric_cols:
        raise ValueError(
            "Input CSV does not contain any of the expected numeric metric columns: "
            + ", ".join(NUMERIC_FIELDS)
        )

    grouped = df.groupby(GROUP_FIELDS, dropna=False)

    agg_metrics = grouped[numeric_cols].agg(["mean", "std"])
    agg_metrics.columns = [f"{stat}_{col}" for col, stat in agg_metrics.columns]
    agg_metrics = agg_metrics.reset_index()

    trial_counts = grouped.size().rename("trial_count").reset_index()
    result = trial_counts.merge(agg_metrics, on=GROUP_FIELDS, how="left")

    # Order columns: grouping fields, counts, then metric pairs (mean/std).
    ordered_cols: list[str] = list(GROUP_FIELDS)
    ordered_cols.append("trial_count")
    for col in numeric_cols:
        mean_col = f"mean_{col}"
        std_col = f"std_{col}"
        if mean_col in result.columns:
            ordered_cols.append(mean_col)
        if std_col in result.columns:
            ordered_cols.append(std_col)

    # Include any remaining columns (e.g., from the merge) at the end.
    remaining = [col for col in result.columns if col not in ordered_cols]
    ordered_cols.extend(remaining)

    result = result[ordered_cols]
    return result.reset_index(drop=True)


def _print_summary_table(df: pd.DataFrame, *, limit: int = 10) -> None:
    if df.empty:
        print("No aggregated rows to display.")
        return

    summary_cols = [
        col
        for col in [
            "param",
            "param_value",
            "shadow",
            "dispatch",
            "n",
            "d",
            "trial_count",
            "mean_savings",
            "std_savings",
            "mean_pooled_pct",
            "std_pooled_pct",
            "mean_ratio_lp",
            "std_ratio_lp",
            "mean_ratio_opt",
            "std_ratio_opt",
            "mean_time_s",
            "std_time_s",
        ]
        if col in df.columns
    ]

    sort_cols = [col for col in ["mean_savings", "mean_pooled_pct", "mean_ratio_lp"] if col in df.columns]
    if sort_cols:
        summary = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    else:
        summary = df

    limit = max(limit, 1)
    to_show = summary.head(limit)
    print(f"\nTop {len(to_show)} configurations (sorted by {', '.join(sort_cols) if sort_cols else 'group order'}):")
    with pd.option_context("display.max_rows", limit, "display.max_columns", len(summary_cols) + 1, "display.float_format", lambda x: f"{x:0.3f}"):
        print(to_show[summary_cols].to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-trial Dynamic Delivery Pooling experiment CSVs into summary statistics."
    )
    parser.add_argument(
        "csv",
        help="Path to a per-trial CSV produced by ddp.scripts.sweep_param or run_many",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out",
        help="Filename for the aggregated CSV (default: append _agg before the extension)",
    )
    parser.add_argument(
        "--limit",
        "-k",
        dest="limit",
        type=int,
        default=10,
        help="Number of summary rows to print to stdout (default: 10)",
    )
    args = parser.parse_args(argv)

    agg_df = aggregate(args.csv)

    if args.out:
        out_path = Path(args.out)
    else:
        inp = Path(args.csv)
        suffix = inp.suffix or ".csv"
        out_path = inp.with_name(inp.stem + "_agg" + suffix)

    _ensure_parent_dir(out_path)
    agg_df.to_csv(out_path, index=False)
    print(f"Wrote aggregated CSV to {out_path.resolve()} ({len(agg_df)} rows)")

    _print_summary_table(agg_df, limit=args.limit)


if __name__ == "__main__":
    main()
