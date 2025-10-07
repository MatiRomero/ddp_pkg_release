#!/usr/bin/env python3
"""Refresh greedy Meituan policy CSVs with updated OPT metrics.

This helper backfills ``ratio_opt`` and ``opt_gap`` in the greedy result dumps
using the baseline policy CSVs as the OPT lookup, then swaps the old greedy rows
in-place.  It mirrors the manual workflow used during the original experiments
but automates the round-trip for every available (day, deadline) pair.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ddp.scripts.merge_policy_results_meituan import (
    _apply_opt_metrics,
    _build_opt_lookup,
)

_DEFAULT_DAYS: Sequence[int] = range(8)
_DEFAULT_DEADLINES: Sequence[int] = (10, 20, 30, 40, 50, 60)
_GREEDY_SHADOWS = {"hd", "pb", "naive", "sh"}
_GREEDY_METHODS = {"greedy", "greedy+"}


def _normalize_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], *, path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _append_missing_opt_total(enriched: pd.DataFrame) -> pd.DataFrame:
    if "opt_total" in enriched.columns:
        return enriched

    savings = pd.to_numeric(enriched.get("savings"), errors="coerce")
    opt_gap = pd.to_numeric(enriched.get("opt_gap"), errors="coerce")
    opt_total = savings + opt_gap
    # Preserve existing columns while inserting the recovered total.
    enriched = enriched.copy()
    enriched["opt_total"] = opt_total
    return enriched


def _ordered_columns(reference: Sequence[str], candidate: Iterable[str]) -> list[str]:
    ordered = list(reference)
    for column in candidate:
        if column not in ordered:
            ordered.append(column)
    return ordered


def refresh_meituan_csv(
    baseline_path: Path,
    greedy_path: Path,
    *,
    dry_run: bool = False,
    backup_suffix: str = ".bak",
) -> bool:
    """Return ``True`` if ``baseline_path`` was updated successfully."""

    if not baseline_path.exists():
        print(f"[skip] missing baseline CSV: {baseline_path}")
        return False

    if not greedy_path.exists():
        print(f"[skip] missing greedy CSV: {greedy_path}")
        return False

    baseline_df = pd.read_csv(baseline_path)
    greedy_df = pd.read_csv(greedy_path)

    _ensure_columns(baseline_df, ("shadow", "method"), path=baseline_path)

    lookup = _build_opt_lookup(baseline_df)
    enriched_greedy = _apply_opt_metrics(greedy_df, lookup)
    enriched_greedy = _append_missing_opt_total(enriched_greedy)

    shadow_mask = _normalize_strings(baseline_df["shadow"]).isin(_GREEDY_SHADOWS)
    method_mask = _normalize_strings(baseline_df["method"]).isin(_GREEDY_METHODS)
    drop_mask = shadow_mask & method_mask

    if drop_mask.any():
        dropped = int(drop_mask.sum())
        print(
            f"[info] removing {dropped} greedy rows from {baseline_path.name} before appending"
        )
    else:
        print(f"[info] no existing greedy rows detected in {baseline_path.name}")

    baseline_pruned = baseline_df.loc[~drop_mask].copy()

    combined = pd.concat([baseline_pruned, enriched_greedy], ignore_index=True, sort=False)
    ordered_columns = _ordered_columns(baseline_pruned.columns, combined.columns)
    combined = combined.loc[:, ordered_columns]

    if dry_run:
        print(
            f"[dry-run] would update {baseline_path} with {len(enriched_greedy)} greedy rows"
        )
        return True

    backup_path = baseline_path.with_suffix(baseline_path.suffix + backup_suffix)
    shutil.copy2(baseline_path, backup_path)
    combined.to_csv(baseline_path, index=False)

    print(
        f"[updated] {baseline_path} (backup: {backup_path.name}, appended {len(enriched_greedy)} rows)"
    )
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the baseline policy CSVs (default: results)",
    )
    parser.add_argument(
        "--greedy-dir",
        type=Path,
        default=Path("greedy_results"),
        help="Directory containing the greedy CSVs (default: greedy_results)",
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs="*",
        default=None,
        help="Specific day indices to refresh (default: 0-7)",
    )
    parser.add_argument(
        "--deadlines",
        type=int,
        nargs="*",
        default=None,
        help="Deadline values to refresh (default: 10 20 30 40 50 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate files without writing updated CSVs",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix used when creating the baseline backup (default: .bak)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    days: Sequence[int] = args.days if args.days is not None else list(_DEFAULT_DAYS)
    deadlines: Sequence[int] = (
        args.deadlines if args.deadlines is not None else list(_DEFAULT_DEADLINES)
    )

    updated_any = False
    for day in days:
        for deadline in deadlines:
            baseline_path = args.results_dir / f"meituan_day{day}_d{deadline}.csv"
            greedy_path = args.greedy_dir / f"meituan_day{day}_d{deadline}_greedys.csv"
            try:
                updated = refresh_meituan_csv(
                    baseline_path,
                    greedy_path,
                    dry_run=args.dry_run,
                    backup_suffix=args.backup_suffix,
                )
            except Exception as exc:  # pragma: no cover - defensive logging helper
                print(f"[error] failed to update {baseline_path.name}: {exc}")
                continue

            updated_any = updated_any or updated

    if not updated_any:
        print("No CSVs were updated — verify the paths and filenames if this is unexpected.")


if __name__ == "__main__":
    main()
