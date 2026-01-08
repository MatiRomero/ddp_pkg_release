# src/ddp/scripts/run_from_config.py
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
from typing import Iterable

# --- repo-path resolver (module-level) ---
_cfg_path = None          # e.g., /.../ddp_pkg_release/configs/config_x.csv
_repo_root = None         # e.g., /.../ddp_pkg_release

def resolve_repo_path(p):
    """
    Resolve paths relative to the repo root (parent of 'configs/').
    - Absolute paths are returned unchanged.
    - Relative paths like 'data/...' or 'results/...' become '/.../ddp_pkg_release/<rel>'.
    - Falsy inputs (None, '') return ''.
    NOTE: _repo_root must be initialized after parsing --config (see snippet #2).
    """
    if not p:
        return ""
    q = pathlib.Path(os.path.expanduser(str(p).strip()))
    if q.is_absolute():
        return str(q)
    if _repo_root is None:
        raise RuntimeError("resolve_repo_path: repo root not initialized; set _repo_root after parsing --config")
    return str((_repo_root / q).resolve())

_COLUMN_TO_FLAG: dict[str, str] = {
    "jobs_csv": "--jobs-csv",
    "jobs": "--jobs",
    "n": "--n",
    "save_csv": "--save_csv",
    "save_job_csv": "--save_job_csv",
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
    "ad_mapping": "--ad-mapping",
    "fix_origin_zero": "--fix-origin-zero",
    "fix-origin-zero": "--fix-origin-zero",
    "flatten_axis": "--flatten-axis",
    "flatten-axis": "--flatten-axis",
    "beta_alpha": "--beta-alpha",
    "beta-alpha": "--beta-alpha",
    "beta_beta": "--beta-beta",
    "beta-beta": "--beta-beta",
}

_BOOLEAN_COLUMNS = {"with_opt", "print_matches", "return_details", "fix_origin_zero", "fix-origin-zero"}
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

    # --- initialize repo root from --config ---
    cfg_path = pathlib.Path(os.path.expanduser(args.config)).resolve()
    repo_root = cfg_path.parent.parent  # parent of 'configs/' => repo root

    # publish to the resolver
    globals()["_cfg_path"]  = cfg_path
    globals()["_repo_root"] = repo_root

    # (optional) normalize args.config itself to absolute path
    args.config = str(cfg_path)

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

    jobs_csv = resolve_repo_path(_normalise(row.get("jobs_csv")))
    jobs_npz = resolve_repo_path(_normalise(row.get("jobs")))
    n_value = _normalise(row.get("n"))
    
    if sum([bool(jobs_csv), bool(jobs_npz), bool(n_value)]) > 1:
        raise SystemExit("Provide only one of 'jobs_csv', 'jobs', or 'n' in the config row")
    
    if jobs_csv:
        job_flag = "--jobs-csv"
        job_value = jobs_csv
    elif jobs_npz:
        job_flag = "--jobs"
        job_value = jobs_npz
    elif n_value:
        job_flag = "--n"
        job_value = n_value
    else:
        raise SystemExit("Config row missing required 'jobs_csv', 'jobs', or 'n' entry")

    save_csv = resolve_repo_path(_normalise(row.get("save_csv")))
    if not save_csv:
        raise SystemExit("Config row missing required 'save_csv' entry")

    job_detail_csv = resolve_repo_path(_normalise(row.get("save_job_csv")))

    d_value = _normalise(row.get("d"))
    if not d_value:
        raise SystemExit("Config row missing required 'd' entry")

    shadow_value = _normalise(row.get("shadows")) or _normalise(row.get("shadow")) or "hd"

    pathlib.Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
    if job_detail_csv:
        pathlib.Path(job_detail_csv).parent.mkdir(parents=True, exist_ok=True)

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

    if job_detail_csv:
        cmd.extend(["--save_job_csv", job_detail_csv])

    for column, value in row.items():
        if column in {"jobs_csv", "jobs", "n", "save_csv", "save_job_csv", "d", "shadow", "shadows"}:
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
