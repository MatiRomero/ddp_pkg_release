"""Merge helper tailored for Meituan-style policy result CSVs.

This script mirrors :mod:`ddp.scripts.merge_policy_results` but relaxes the
schema assumptions to match the one-off CSVs produced for the Meituan
experiments.  Those files lack the sweep metadata columns (``param``,
``param_value``, ``trial_index``) but still contain enough information to align
trials via the trio ``(n, d, seed)``.  The helper appends the new policy rows
to the existing CSV and backfills ``opt_total``, ``opt_gap``, and ``ratio_opt``
for the appended entries using the optimum totals recorded in the baseline
file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Meituan result CSVs identify a trial via ``(n, d, seed)``.
TRIAL_KEY_COLUMNS: Sequence[str] = ("n", "d", "seed")

_OPT_TOTAL_RTOL = 1e-6
_OPT_TOTAL_ATOL = 1e-8


def _require_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _normalized_trial_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the trial key columns coerced to canonical dtypes."""

    keys = df.loc[:, TRIAL_KEY_COLUMNS].copy()
    for column in TRIAL_KEY_COLUMNS:
        if column not in keys.columns:
            continue

        keys[column] = pd.to_numeric(keys[column], errors="coerce").astype("Float64")

    return keys


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") if series is not None else series


def _derive_opt_total(df: pd.DataFrame) -> pd.Series:
    """Return a Series containing the total OPT value for each row."""

    opt_total = pd.Series(pd.NA, index=df.index, dtype="Float64")

    if "opt_total" in df.columns:
        opt_total = _coerce_numeric(df["opt_total"]).astype("Float64")

    if "opt_gap" in df.columns:
        savings = _coerce_numeric(df.get("savings", pd.Series(index=df.index))).astype("Float64")
        opt_gap = _coerce_numeric(df["opt_gap"]).astype("Float64")
        derived = savings + opt_gap
        opt_total = opt_total.fillna(derived)

    if "ratio_opt" in df.columns:
        savings = _coerce_numeric(df.get("savings", pd.Series(index=df.index))).astype("Float64")
        ratio_opt = _coerce_numeric(df["ratio_opt"]).astype("Float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            derived = savings.divide(ratio_opt)
        derived = derived.where(~(ratio_opt == 0))
        opt_total = opt_total.fillna(derived)

    opt_indicator = None
    for column in ("dispatch", "shadow", "method"):
        if column in df.columns:
            column_mask = df[column].astype("string").str.lower() == "opt"
            opt_indicator = column_mask if opt_indicator is None else (opt_indicator | column_mask)

    if opt_indicator is not None:
        savings = _coerce_numeric(df.get("savings", pd.Series(index=df.index))).astype("Float64")
        opt_total = opt_total.mask(opt_indicator, savings)
        opt_total = opt_total.fillna(savings.where(opt_indicator))

    return opt_total


def _build_opt_lookup(existing: pd.DataFrame) -> pd.Series:
    """Construct a lookup Series keyed by the trial identifier columns."""

    _require_columns(existing, TRIAL_KEY_COLUMNS, label="Existing CSV")

    opt_total = _derive_opt_total(existing)
    if opt_total.isna().all():
        raise ValueError(
            "Existing CSV does not contain opt_total, opt_gap, or ratio_opt values "
            "needed to recover the optimum totals."
        )

    key_frame = _normalized_trial_keys(existing)
    enriched = existing.assign(_opt_total=opt_total)
    for column in TRIAL_KEY_COLUMNS:
        enriched[column] = key_frame[column]

    representatives = []
    conflicts = []
    grouped = enriched.groupby(list(TRIAL_KEY_COLUMNS), dropna=False)["_opt_total"]
    for key, values in grouped:
        non_null = values.dropna()
        if non_null.empty:
            continue

        numeric = non_null.astype("float64")
        reference = numeric.iloc[0]
        close_mask = np.isclose(
            numeric.to_numpy(),
            reference,
            rtol=_OPT_TOTAL_RTOL,
            atol=_OPT_TOTAL_ATOL,
        )
        if not close_mask.all():
            conflicts.append(key)
            continue

        representative = float(numeric.mean())
        representatives.append((*key, representative))

    if conflicts:
        conflict_df = pd.DataFrame(conflicts, columns=list(TRIAL_KEY_COLUMNS))
        raise ValueError(
            "Existing CSV has conflicting opt_total values for some trial keys "
            "beyond the allowed tolerance. Resolve the duplicates before merging.\n"
            + conflict_df.to_string(index=False)
        )

    if not representatives:
        raise ValueError("Unable to build OPT lookup from the existing CSV.")

    lookup_df = pd.DataFrame(representatives, columns=[*TRIAL_KEY_COLUMNS, "_opt_total"])
    lookup = lookup_df.set_index(list(TRIAL_KEY_COLUMNS))["_opt_total"].astype("Float64")
    return lookup


def _apply_opt_metrics(new_rows: pd.DataFrame, lookup: pd.Series) -> pd.DataFrame:
    """Fill opt_total, opt_gap, and ratio_opt for ``new_rows`` in-place."""

    _require_columns(new_rows, TRIAL_KEY_COLUMNS, label="New CSV")

    new_rows = new_rows.copy()
    has_opt_total = "opt_total" in new_rows.columns
    key_index = pd.MultiIndex.from_frame(_normalized_trial_keys(new_rows))
    opt_from_lookup = lookup.reindex(key_index)

    if opt_from_lookup.isna().any():
        missing_mask = opt_from_lookup.isna().to_numpy()
        missing = new_rows.loc[missing_mask, TRIAL_KEY_COLUMNS]
        missing_unique = missing.drop_duplicates()

        hint = ""
        if missing_mask.all():
            hint = (
                "\nNo matching trial keys were found in the existing CSV. "
                "Double-check that the first CLI argument points to the baseline "
                "file containing OPT totals and the second argument is the new "
                "policy results to append."
            )

        raise ValueError(
            "Missing OPT totals for the following trial keys:\n"
            + missing_unique.to_string(index=False)
            + hint
        )

    existing_opt_series = new_rows.get("opt_total")
    if existing_opt_series is None:
        opt_total_existing = pd.Series(pd.NA, index=new_rows.index, dtype="Float64")
    else:
        opt_total_existing = _coerce_numeric(existing_opt_series).astype("Float64")

    opt_total_lookup = opt_from_lookup.reset_index(drop=True)
    opt_total_lookup.index = new_rows.index
    opt_total_lookup = opt_total_lookup.astype("Float64")

    opt_total_filled = opt_total_existing.combine_first(opt_total_lookup).astype("Float64")

    savings = _coerce_numeric(new_rows.get("savings", pd.Series(index=new_rows.index))).astype("Float64")

    opt_gap = opt_total_filled - savings
    opt_gap = opt_gap.clip(lower=0)
    ratio_opt = savings.divide(opt_total_filled)
    ratio_opt = ratio_opt.where(opt_total_filled != 0)

    if has_opt_total:
        new_rows["opt_total"] = opt_total_filled
    new_rows["opt_gap"] = opt_gap
    new_rows["ratio_opt"] = ratio_opt

    return new_rows


def merge_policy_results(existing_path: Path, new_path: Path, output_path: Path) -> Path:
    existing_df = pd.read_csv(existing_path)
    new_df = pd.read_csv(new_path)

    opt_lookup = _build_opt_lookup(existing_df)
    new_df_with_opt = _apply_opt_metrics(new_df, opt_lookup)

    if not new_df_with_opt.empty:
        all_na_columns = [col for col in new_df_with_opt.columns if new_df_with_opt[col].isna().all()]
        if all_na_columns:
            new_df_with_opt = new_df_with_opt.drop(columns=all_na_columns)

    combined = pd.concat([existing_df, new_df_with_opt], ignore_index=True, sort=False)

    if output_path.exists() and output_path.resolve() == existing_path.resolve():
        backup_path = existing_path.with_suffix(existing_path.suffix + ".bak")
        combined.to_csv(backup_path, index=False)

    combined.to_csv(output_path, index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("existing", type=Path, help="Path to the existing CSV containing baseline policies.")
    parser.add_argument("new", type=Path, help="Path to the CSV with the new policy results to append.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination for the merged CSV. Defaults to overwriting the existing file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    existing_path = args.existing.expanduser().resolve()
    new_path = args.new.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else existing_path

    for path, label in [(existing_path, "existing"), (new_path, "new")]:
        if not path.exists():
            raise FileNotFoundError(f"The {label} CSV does not exist: {path}")

    merged_path = merge_policy_results(existing_path, new_path, output_path)
    print(f"Merged CSV written to {merged_path}")


if __name__ == "__main__":
    main()

