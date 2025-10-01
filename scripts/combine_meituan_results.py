"""Combine per-day Meituan sweep CSVs into a single per-trial table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


MEITUAN_FILENAME_RE = re.compile(r"day(?P<day>\d+)_d(?P<d>\d+)")


def _expand_inputs(patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.exists():
            files.append(path)
            continue
        parent = path.parent if path.parent != Path("") else Path.cwd()
        files.extend(sorted(parent.glob(path.name)))
    unique_files: dict[Path, None] = {}
    for file in files:
        resolved = file.resolve()
        if resolved not in unique_files:
            unique_files[resolved] = None
    return sorted(unique_files.keys())


def combine(inputs: Iterable[str]) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []
    for item in _expand_inputs(inputs):
        if not item.exists():
            raise FileNotFoundError(f"Input CSV not found: {item}")
        df = pd.read_csv(item)

        match = MEITUAN_FILENAME_RE.search(item.stem)
        if match and "day" not in df.columns:
            df["day"] = int(match.group("day"))
        if match and "d" in match.groupdict():
            d_value = float(match.group("d"))
            if "d" not in df.columns:
                df["d"] = d_value
            else:
                df["d"] = pd.to_numeric(df["d"], errors="coerce").fillna(d_value)
        else:
            df["d"] = pd.to_numeric(df.get("d"), errors="coerce")

        df["param"] = "d"
        df["param_value"] = pd.to_numeric(df["d"], errors="coerce")
        dataframes.append(df)

    if not dataframes:
        raise ValueError("No input CSV files were found.")

    combined = pd.concat(dataframes, ignore_index=True, sort=False)

    sort_keys: list[str] = []
    for key in ("d", "day", "seed"):
        if key in combined.columns:
            sort_keys.append(key)
    if sort_keys:
        combined = combined.sort_values(sort_keys, na_position="last")

    # Ensure CSV writer doesn't inject Windows newlines by using lineterminator.
    combined = combined.reset_index(drop=True)
    return combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Meituan per-day sweep CSVs into a single per-trial file with"
            " param/param_value columns for downstream aggregation."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input CSV paths or glob patterns (e.g., results/meituan_day*_d*.csv)",
    )
    parser.add_argument(
        "--out",
        "-o",
        required=True,
        help="Filename for the combined CSV",
    )
    args = parser.parse_args(argv)

    combined = combine(args.inputs)

    if "n" in combined.columns:
        combined = combined.drop(columns=["n"])

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
