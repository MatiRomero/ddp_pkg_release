"""Run ``meituan_average_duals`` tasks from a CSV config."""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
from typing import Sequence

REQUIRED_COLUMNS = (
    "day",
    "deadline",
    "resolution",
    "data_dir",
    "jobs_pattern",
    "export_dir",
)


def _load_rows(config_path: str) -> list[dict[str, str]]:
    with open(config_path, newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No rows found in {config_path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise SystemExit(
            f"Config {config_path} is missing required columns: {', '.join(missing)}"
        )
    return rows


def _select_row(rows: Sequence[dict[str, str]]) -> tuple[int, dict[str, str]]:
    sge_task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    index = sge_task_id - 1
    if not (0 <= index < len(rows)):
        raise SystemExit(
            f"SGE_TASK_ID={sge_task_id} out of range for {len(rows)} rows"
        )
    return sge_task_id, rows[index]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run meituan_average_duals from a config CSV")
    parser.add_argument("--config", default="configs/ad_config.csv", help="Path to the config CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command instead of executing it",
    )
    return parser.parse_known_args()



def main() -> None:
    args, forward = parse_args()
    rows = _load_rows(args.config)
    sge_task_id, row = _select_row(rows)

    export_dir = row["export_dir"].strip()
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ddp.scripts.meituan_average_duals",
        *forward,
        "--day",
        row["day"].strip(),
        "--deadline",
        row["deadline"].strip(),
        "--resolution",
        row["resolution"].strip(),
        "--data-dir",
        row["data_dir"].strip(),
        "--jobs-pattern",
        row["jobs_pattern"].strip(),
        "--export-dir",
        export_dir,
    ]

    print(f"[INFO] SGE_TASK_ID={sge_task_id} -> row {sge_task_id}/{len(rows)}")
    print("[INFO] Running:", " ".join(cmd))
    if args.dry_run:
        return

    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
