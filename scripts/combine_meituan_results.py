"""Combine per-day Meituan sweep CSVs into a single per-trial table."""

from __future__ import annotations

import argparse
from pathlib import Path

from ddp.results.combine import combine_meituan_results


combine = combine_meituan_results


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

    combined = combine_meituan_results(args.inputs)

    if "n" in combined.columns:
        combined = combined.drop(columns=["n"])

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
