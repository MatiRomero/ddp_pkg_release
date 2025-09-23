"""Aggregate hindsight-dual datasets into average-dual lookup tables."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable

from ddp.scripts.average_duals import _load_mapping, MappingCallable


@dataclass
class _Stats:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ZeroDivisionError("Cannot compute mean with zero count")
        return self.total / self.count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hd_csv",
        type=Path,
        help="Path to the hindsight-dual dataset produced by build_hd_dataset.py",
    )
    parser.add_argument(
        "mapping",
        help="module:callable specification for the coordinate-based type mapping",
    )
    parser.add_argument(
        "out_csv",
        type=Path,
        help="Destination CSV file for the aggregated average-dual table",
    )
    return parser


def _iter_hd_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Hindsight-dual CSV must include a header row")
        required = {
            "origin_x",
            "origin_y",
            "dest_x",
            "dest_y",
            "hindsight_dual",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Hindsight-dual CSV missing required columns: {missing_list}")
        yield from reader


def compute_average_duals(csv_path: Path, mapping: MappingCallable) -> dict[Hashable, _Stats]:
    stats: dict[Hashable, _Stats] = {}
    for row in _iter_hd_rows(csv_path):
        try:
            origin_x = float(row["origin_x"])
            origin_y = float(row["origin_y"])
            dest_x = float(row["dest_x"])
            dest_y = float(row["dest_y"])
            dual = float(row["hindsight_dual"])
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid numeric value in hindsight-dual CSV") from exc

        key = mapping(origin_x, origin_y, dest_x, dest_y)
        bucket = stats.get(key)
        if bucket is None:
            bucket = stats[key] = _Stats()
        bucket.add(dual)
    return stats


def write_average_duals(csv_path: Path, stats: dict[Hashable, _Stats]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["type", "mean_dual", "count"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(stats.keys(), key=repr):
            bucket = stats[key]
            writer.writerow(
                {
                    "type": str(key),
                    "mean_dual": f"{bucket.mean:.12g}",
                    "count": bucket.count,
                }
            )


def _report_coverage(stats: dict[Hashable, _Stats], expected: set[Hashable] | None) -> None:
    if not expected:
        return
    missing = [key for key in expected if key not in stats]
    if not missing:
        return
    print(
        "Warning: mapping expected %d types but %d were absent from the dataset"
        % (len(expected), len(missing))
    )
    preview = sorted(missing, key=repr)[:10]
    for key in preview:
        print(f"  Missing: {key!r}")
    if len(missing) > len(preview):
        remaining = len(missing) - len(preview)
        print(f"  ... and {remaining} more")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    mapping, expected = _load_mapping(args.mapping)
    stats = compute_average_duals(args.hd_csv, mapping)
    write_average_duals(args.out_csv, stats)
    print(
        "Wrote %d average-dual entries to %s"
        % (len(stats), args.out_csv)
    )
    _report_coverage(stats, expected)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
