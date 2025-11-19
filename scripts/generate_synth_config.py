"""Generate configuration CSVs for synthetic job archives.

This helper enumerates seeds and ``d`` values, pairing each seed with a
pre-generated ``.npz`` instance. The resulting CSV feeds directly into
``ddp.scripts.run_from_config`` so cluster launches can select rows via
``$SGE_TASK_ID``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


_DEFAULT_SHADOWS: Sequence[str] = ["naive", "pb", "hd", "ad"]


def _parse_ints(payload: str) -> list[int]:
    return [int(token.strip()) for token in payload.replace(";", ",").split(",") if token.strip()]


def _parse_floats(payload: str) -> list[float]:
    return [float(token.strip()) for token in payload.replace(";", ",").split(",") if token.strip()]


def _sanitise(value: object) -> str:
    return str(value).strip().replace("/", "_")


def _iter_rows(
    *,
    seeds: Iterable[int],
    d_values: Iterable[float],
    shadows: Iterable[str],
    jobs_pattern: str,
    dispatch: str,
    results_dir: Path,
    include_job_details: bool,
) -> Iterable[dict[str, str]]:
    for seed in seeds:
        jobs_path = Path(jobs_pattern.format(seed=seed))
        for d in d_values:
            for shadow in shadows:
                base_name = f"seed{seed}_d{_sanitise(d)}_{shadow}_{dispatch}"
                save_path = results_dir / f"{base_name}.csv"
                job_details_path = results_dir / f"{base_name}_jobs.csv"
                row = {
                    "jobs": str(jobs_path),
                    "d": str(d),
                    "shadows": shadow,
                    "dispatch": dispatch,
                    "seed": str(seed),
                    "save_csv": str(save_path),
                    "save_job_csv": str(job_details_path) if include_job_details else "",
                }
                yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Label for the generated CSV and results directory (defaults to timestamp)",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=Path("configs"),
        help="Directory to store the generated config CSV",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results") / "synth_runs",
        help="Base directory for run outputs (a subdir using --config-name is created)",
    )
    parser.add_argument(
        "--jobs-pattern",
        default="data/synth_seed{seed}.npz",
        help="Path template for the synthetic job archives (must include {seed})",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma- or semicolon-separated seeds to enumerate",
    )
    parser.add_argument(
        "--d-values",
        default="30,60,90",
        help="Comma- or semicolon-separated d values to sweep",
    )
    parser.add_argument(
        "--shadows",
        default=",".join(_DEFAULT_SHADOWS),
        help="Comma-separated list of shadow policies",
    )
    parser.add_argument(
        "--dispatch",
        default="batch",
        help="Dispatch policy for every row",
    )
    parser.add_argument(
        "--no-job-details",
        dest="include_job_details",
        action="store_false",
        help="Leave the save_job_csv column empty",
    )
    parser.set_defaults(include_job_details=True)

    args = parser.parse_args()

    configs_dir: Path = args.configs_dir
    configs_dir.mkdir(parents=True, exist_ok=True)

    results_dir: Path = Path(args.results_dir) / args.config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_ints(args.seeds)
    d_values = _parse_floats(args.d_values)
    shadows = [token.strip() for token in args.shadows.split(",") if token.strip()]

    rows = list(
        _iter_rows(
            seeds=seeds,
            d_values=d_values,
            shadows=shadows,
            jobs_pattern=args.jobs_pattern,
            dispatch=args.dispatch,
            results_dir=results_dir,
            include_job_details=args.include_job_details,
        )
    )

    config_path = configs_dir / f"config_synth_{args.config_name}.csv"
    headers = [
        "jobs",
        "d",
        "shadows",
        "dispatch",
        "seed",
        "save_csv",
        "save_job_csv",
    ]

    with config_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {config_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
