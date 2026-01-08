"""Generate hindsight-dual datasets by sampling LP relaxations."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np

from ddp.engine.opt import compute_lp_relaxation
from ddp.model import generate_jobs, Job
from ddp.scripts.run import reward_fn


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help="Number of independently sampled instances to generate (default: 1).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of jobs per instance (default: 50).",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=3.0,
        help="Scalar slack / deadline parameter passed to the LP relaxation (default: 3.0).",
    )
    parser.add_argument(
        "--seed0",
        type=int,
        default=0,
        help="Base seed; instance i uses rng seed (seed0 + i).",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Destination CSV file for the generated dataset.",
    )
    parser.add_argument(
        "--fix-origin-zero",
        action="store_true",
        help="Set every generated job origin to the depot at (0, 0)",
    )
    parser.add_argument(
        "--flatten-axis",
        choices=["x", "y"],
        help="Project all jobs onto a single axis by zeroing the chosen coordinate",
    )
    parser.add_argument(
        "--beta-alpha",
        type=float,
        default=1.0,
        help="Alpha parameter for Beta distribution (default: 1.0, which gives uniform distribution).",
    )
    parser.add_argument(
        "--beta-beta",
        type=float,
        default=1.0,
        help="Beta parameter for Beta distribution (default: 1.0, which gives uniform distribution).",
    )
    return parser.parse_args()


def _write_rows(csv_path: Path, rows: list[dict[str, float | int]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instance_id",
        "job_index",
        "seed",
        "timestamp",
        "origin_x",
        "origin_y",
        "dest_x",
        "dest_y",
        "d",
        "potential",
        "hindsight_dual",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    n_instances = int(args.instances)
    if n_instances <= 0:
        raise ValueError("--instances must be positive")

    n_jobs = int(args.n)
    if n_jobs <= 1:
        raise ValueError("--n must exceed one to form pairings")

    deadline = float(args.d)

    rows: list[dict[str, float | int]] = []
    solve_times: list[float] = []

    for instance_id in range(n_instances):
        seed = int(args.seed0) + instance_id
        rng = np.random.default_rng(seed)
        jobs = generate_jobs(n_jobs, rng, beta_alpha=args.beta_alpha, beta_beta=args.beta_beta)

        # Apply fix_origin_zero if requested
        if args.fix_origin_zero:
            jobs = [Job(origin=(0.0, 0.0), dest=job.dest, timestamp=job.timestamp) for job in jobs]

        # Apply flatten_axis if requested
        if args.flatten_axis is not None:
            axis = 0 if args.flatten_axis == "x" else 1
            
            def _flatten(point: tuple[float, float]) -> tuple[float, float]:
                coords = [float(point[0]), float(point[1])]
                coords[axis] = 0.0
                return coords[0], coords[1]
            
            jobs = [
                Job(origin=_flatten(job.origin), dest=_flatten(job.dest), timestamp=job.timestamp)
                for job in jobs
            ]

        start = time.perf_counter()
        lp_result = compute_lp_relaxation(jobs, reward_fn, time_window=deadline)
        elapsed = time.perf_counter() - start
        solve_times.append(elapsed)

        duals = list(map(float, lp_result["duals"]))
        if len(duals) != n_jobs:
            msg = "LP returned a dual vector with unexpected length"
            raise RuntimeError(msg)

        for job_index, (job, dual) in enumerate(zip(jobs, duals)):
            rows.append(
                {
                    "instance_id": instance_id,
                    "job_index": job_index,
                    "seed": seed,
                    "timestamp": float(job.timestamp),
                    "origin_x": float(job.origin[0]),
                    "origin_y": float(job.origin[1]),
                    "dest_x": float(job.dest[0]),
                    "dest_y": float(job.dest[1]),
                    "d": deadline,
                    "potential": 0.5 * float(job.length),
                    "hindsight_dual": dual,
                }
            )

    _write_rows(Path(args.out_csv), rows)

    total_time = sum(solve_times)
    mean_time = statistics.fmean(solve_times) if solve_times else 0.0
    min_time = min(solve_times) if solve_times else 0.0
    max_time = max(solve_times) if solve_times else 0.0
    print(
        "Generated %d instances (n=%d) -> %s" % (n_instances, n_jobs, Path(args.out_csv))
    )
    print(
        "LP solve time (s): mean=%.4f  min=%.4f  max=%.4f  total=%.4f"
        % (mean_time, min_time, max_time, total_time)
    )


if __name__ == "__main__":
    main()
