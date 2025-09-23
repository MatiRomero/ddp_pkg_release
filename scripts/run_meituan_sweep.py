"""Batch runner for Meituan lunchtime datasets.

Run this script from the repository root (``python scripts/run_meituan_sweep.py``)
with the package installed or the virtual environment activated so that
``ddp`` is importable.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


D_VALUES = [10, 20, 30]
DAYS = range(8)


def run_sweep() -> None:
    """Execute the SHADOW×DISPATCH CLI over the configured parameter sweep."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path.cwd()

    for day in DAYS:
        jobs_csv = repo_root / "data" / f"meituan_city_lunchtime_plat10301330_day{day}.csv"
        export_npz = repo_root / "data" / f"meituan_day{day}.npz"
        if not jobs_csv.exists():
            LOGGER.warning("Skipping day %s because %s is missing", day, jobs_csv)
            continue

        for d in D_VALUES:
            results_csv = repo_root / "results" / f"meituan_day{day}_d{d}.csv"
            results_csv.parent.mkdir(parents=True, exist_ok=True)

            command = [
                sys.executable,
                "-m",
                "ddp.scripts.run",
                "--with_opt",
                "--jobs-csv",
                str(jobs_csv),
                "--timestamp-column",
                "platform_order_time",
                "--export-npz",
                str(export_npz),
                "--save_csv",
                str(results_csv),
                "--d",
                str(d),
            ]

            LOGGER.info("Running day=%s d=%s via ddp.scripts.run: %s", day, d, command)

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as exc:
                LOGGER.error(
                    "Command failed for day=%s d=%s (return code %s)",
                    day,
                    d,
                    exc.returncode,
                    exc_info=exc,
                )


def main() -> None:
    """Entry point for command-line execution."""

    run_sweep()


if __name__ == "__main__":
    main()
