"""Generate CSV configs for batch average-dual runs."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_DAYS: Sequence[str] = (
    "0",
    "1",
    # "2",
    # "3",
    # "4",
    # "5",
    # "6",
    # "7",
)
# DEFAULT_DEADLINES: Sequence[str] = ("90","120","150","180","210","240","270","300")
DEFAULT_DEADLINES: Sequence[str] = ("5","15")
DEFAULT_RESOLUTIONS: Sequence[str] = ("7", "8", "9", "10")
DEFAULT_DATA_DIR = "data"
DEFAULT_JOBS_PATTERN = "meituan_area6_lunchtime_plat10301330_day{day}.csv"
DEFAULT_EXPORT_DIR = "data/average_duals_area6"


def _parse_sequence(value: Iterable[str]) -> list[str]:
    return [str(item) for item in value]


def build_rows(
    *,
    days: Sequence[str],
    deadlines: Sequence[str],
    resolutions: Sequence[str],
    data_dir: str,
    jobs_pattern: str,
    export_dir: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for day, deadline, resolution in product(days, deadlines, resolutions):
        rows.append(
            {
                "day": str(day),
                "deadline": str(deadline),
                "resolution": str(resolution),
                "data_dir": str(data_dir),
                "jobs_pattern": str(jobs_pattern),
                "export_dir": str(export_dir),
            }
        )
    return rows


def write_config(path: Path, rows: Sequence[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("day", "deadline", "resolution", "data_dir", "jobs_pattern", "export_dir"),
        )
        writer.writeheader()
        writer.writerows(rows)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an average-dual config CSV")
    parser.add_argument(
        "--config",
        default="configs/ad_config.csv",
        help="Destination CSV path",
    )
    parser.add_argument(
        "--days",
        nargs="*",
        default=DEFAULT_DAYS,
        help="List of days to process",
    )
    parser.add_argument(
        "--deadlines",
        nargs="*",
        default=DEFAULT_DEADLINES,
        help="LP deadlines (seconds)",
    )
    parser.add_argument(
        "--resolutions",
        nargs="*",
        default=DEFAULT_RESOLUTIONS,
        help="H3 resolutions",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory containing snapshot CSVs",
    )
    parser.add_argument(
        "--jobs-pattern",
        default=DEFAULT_JOBS_PATTERN,
        help="Filename pattern used to locate each snapshot (must include {day})",
    )
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Directory for average-dual exports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated rows without writing the CSV",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    days = _parse_sequence(args.days)
    deadlines = _parse_sequence(args.deadlines)
    resolutions = _parse_sequence(args.resolutions)

    rows = build_rows(
        days=days,
        deadlines=deadlines,
        resolutions=resolutions,
        data_dir=str(args.data_dir),
        jobs_pattern=str(args.jobs_pattern),
        export_dir=str(args.export_dir),
    )

    if args.dry_run:
        for row in rows:
            print(
                f"day={row['day']} deadline={row['deadline']} resolution={row['resolution']} "
                f"data_dir={row['data_dir']} jobs_pattern={row['jobs_pattern']} export_dir={row['export_dir']}"
            )
        return

    write_config(Path(args.config), rows)
    print(f"[INFO] Wrote {len(rows)} rows to {args.config}")


if __name__ == "__main__":
    main()
