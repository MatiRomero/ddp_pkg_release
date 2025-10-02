"""Batch merge Meituan policy result CSVs with their AD counterparts.

Usage:
    python scripts/batch_merge_meituan.py [--results-dir results]

The script scans days 0-7 and d values (10, 20, 30, 40, 50, 60), invoking
``python -m ddp.scripts.merge_policy_results_meituan`` for each pair of
`results/meituan_day{day}_d{d}.csv` and the corresponding `_ad` file. Missing
files are reported and skipped so a single command can merge every available
combination.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

D_VALUES = (10, 20, 30, 40, 50, 60)


def merge_all(results_dir: Path) -> None:
    """Merge all available Meituan CSV pairs within ``results_dir``."""

    for day in range(8):
        for d in D_VALUES:
            base = results_dir / f"meituan_day{day}_d{d}.csv"
            ad = results_dir / f"meituan_day{day}_d{d}_ad.csv"
            output = results_dir / f"meituan_day{day}_d{d}_withAD.csv"

            missing = [path for path in (base, ad) if not path.exists()]
            if missing:
                missing_str = ", ".join(str(path) for path in missing)
                print(
                    f"Skipping day {day}, d {d}: missing required file(s): {missing_str}"
                )
                continue

            cmd = [
                "python",
                "-m",
                "ddp.scripts.merge_policy_results_meituan",
                str(base),
                str(ad),
                "--output",
                str(output),
            ]

            print(f"Merging day {day}, d {d}: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                print(
                    f"Error merging day {day}, d {d} (return code {exc.returncode})."
                )
            except FileNotFoundError as exc:
                print(
                    f"Failed to execute merge command for day {day}, d {d}: {exc}."
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch merge Meituan policy results with AD policy outputs."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        type=Path,
        help="Directory containing Meituan CSV files (default: results)",
    )
    args = parser.parse_args(argv)

    merge_all(args.results_dir)


if __name__ == "__main__":
    main()
