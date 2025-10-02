"""Convert Area 6 summary CSV output into the canonical shadow sweep schema.

Workflow
========
1. Run :mod:`ddp.scripts.meituan_area6_shadow_sweep` to aggregate the gamma/tau
   grid across the eight Meituan Area 6 lunch periods.  The script writes a
   compact summary CSV with per-configuration means and standard deviations.
2. Use this module as a command line utility to translate the summary CSV into
   the "dataset" geometry format emitted by :mod:`ddp.scripts.shadow_sweep`.
   The resulting file can be passed directly to
   :mod:`ddp.scripts.plot_shadow_sweep` (or any tooling that expects the
   canonical schema) without rerunning the expensive sweep.

The conversion rewrites each row to include the canonical `geometry`,
`metric`, `mean`, `std`, and trial counts while preserving the original `d`
window and H3 `resolution` as optional passthrough columns for historical
context.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


_INPUT_COLUMNS = {
    "d",
    "dispatch",
    "shadow",
    "gamma",
    "tau",
    "resolution",
    "metric_mean",
    "metric_std",
}


_CANONICAL_COLUMNS = [
    "geometry",
    "shadow",
    "gamma",
    "tau",
    "dispatch",
    "metric",
    "mean",
    "std",
    "trials",
    "valid_trials",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to Area 6 summary CSV")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination where the canonical CSV will be written",
    )
    parser.add_argument(
        "--metric",
        default="savings",
        help="Metric label to store in the canonical CSV (default: savings)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=8,
        help="Total number of trials represented by the summary (default: 8)",
    )
    parser.add_argument(
        "--valid-trials",
        dest="valid_trials",
        type=int,
        default=8,
        help="Number of valid trials represented by the summary (default: 8)",
    )
    return parser


def _convert_row(row: dict[str, str], metric: str, trials: int, valid_trials: int) -> dict[str, object]:
    try:
        mean_value = float(row["metric_mean"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"invalid metric_mean in row: {row}") from exc

    std_raw = row.get("metric_std", "")
    std_value: float | str
    if std_raw == "" or std_raw is None:
        std_value = ""
    else:
        try:
            std_value = float(std_raw)
        except ValueError as exc:
            raise ValueError(f"invalid metric_std in row: {row}") from exc

    converted: dict[str, object] = {
        "geometry": "dataset",
        "shadow": row["shadow"],
        "gamma": row["gamma"],
        "tau": row["tau"],
        "dispatch": row["dispatch"],
        "metric": metric,
        "mean": mean_value,
        "std": std_value,
        "trials": trials,
        "valid_trials": valid_trials,
    }

    for passthrough in ("d", "resolution"):
        if passthrough in row:
            converted[passthrough] = row[passthrough]

    return converted


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"input CSV not found: {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            parser.error("input CSV is missing a header row")

        missing = _INPUT_COLUMNS - set(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            parser.error(
                "input CSV is missing required column(s): " f"{missing_list}"
            )

        passthrough_columns = [
            column
            for column in ("d", "resolution")
            if column in reader.fieldnames
        ]
        fieldnames = _CANONICAL_COLUMNS + passthrough_columns

        with output_path.open("w", newline="", encoding="utf-8") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                converted = _convert_row(
                    row,
                    metric=args.metric,
                    trials=args.trials,
                    valid_trials=args.valid_trials,
                )
                writer.writerow(converted)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
