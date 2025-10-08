# src/ddp/scripts/run_from_config.py
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
from typing import Iterable


_COLUMN_TO_FLAG: dict[str, str] = {
    "jobs_csv": "--jobs-csv",
    "jobs": "--jobs",
    "save_csv": "--save_csv",
    "d": "--d",
    "shadow": "--shadows",
    "shadows": "--shadows",
    "dispatch": "--dispatch",
    "gamma": "--gamma",
    "tau": "--tau",
    "plus_gamma": "--plus_gamma",
    "plus_tau": "--plus_tau",
    "seed": "--seed",
    "timestamp_column": "--timestamp-column",
    "export_npz": "--export-npz",
    "with_opt": "--with_opt",
    "opt_method": "--opt_method",
    "print_matches": "--print_matches",
    "return_details": "--return_details",
    "tie_breaker": "--tie_breaker",
    "ad_duals": "--ad_duals",
    "ad_resolution": "--ad_resolution",
    "ad_resolutions": "--ad_resolutions",
    "ad_mapping": "--ad_mapping",
}

_BOOLEAN_COLUMNS = {"with_opt", "print_matches", "return_details"}
_REPEATABLE_COLUMNS = {"ad_resolution"}


def _normalise(value: str | None) -> str:
    return value.strip() if value else ""


def _split_repeatable(value: str) -> Iterable[str]:
    cleaned = value.replace(";", ",")
    for part in cleaned.split(","):
        token = part.strip()
        if token:
            yield token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.csv", help="Path to config CSV")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command and exit")
    # Everything else (fixed flags) is forwarded to ddp.scripts.run:
    args, forward = parser.parse_known_args()

    sge_task_id = int(os.environ.get("SGE_TASK_ID", "1"))
    row_idx = sge_task_id - 1

    with open(args.config, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"No rows found in {args.config}")
    if not (0 <= row_idx < len(rows)):
        raise SystemExit(
            f"SGE_TASK_ID={sge_task_id} out of range for {len(rows)} rows in {args.config}"
        )

    row = rows[row_idx]

    jobs_csv = _normalise(row.get("jobs_csv"))
    jobs_npz = _normalise(row.get("jobs"))
    if jobs_csv and jobs_npz:
        raise SystemExit("Provide only one of 'jobs_csv' or 'jobs' in the config row")
    if jobs_csv:
        job_flag = "--jobs-csv"
        job_value = jobs_csv
    elif jobs_npz:
        job_flag = "--jobs"
        job_value = jobs_npz
    else:
        raise SystemExit("Config row missing required 'jobs_csv' or 'jobs' entry")

    save_csv = _normalise(row.get("save_csv"))
    if not save_csv:
        raise SystemExit("Config row missing required 'save_csv' entry")

    d_value = _normalise(row.get("d"))
    if not d_value:
        raise SystemExit("Config row missing required 'd' entry")

    shadow_value = _normalise(row.get("shadows")) or _normalise(row.get("shadow")) or "hd"

    pathlib.Path(save_csv).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "ddp.scripts.run",
        *forward,
        job_flag,
        job_value,
        "--save_csv",
        save_csv,
        "--d",
        d_value,
        "--shadows",
        shadow_value,
    ]

    for column, value in row.items():
        if column in {"jobs_csv", "jobs", "save_csv", "d", "shadow", "shadows"}:
            continue
        flag = _COLUMN_TO_FLAG.get(column)
        if not flag:
            continue

        normalised = _normalise(value)
        if not normalised:
            continue

        if column in _BOOLEAN_COLUMNS:
            if normalised.lower() in {"1", "true", "yes", "y"}:
                cmd.append(flag)
            continue

        if column in _REPEATABLE_COLUMNS:
            for entry in _split_repeatable(normalised):
                cmd.extend([flag, entry])
            continue

        cmd.extend([flag, normalised])

    print(f"[INFO] SGE_TASK_ID={sge_task_id} -> row {row_idx + 1}/{len(rows)}")
    print("[INFO] Running:", " ".join(cmd))
    if args.dry_run:
        return

    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
