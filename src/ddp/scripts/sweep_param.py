"""Unified parameter sweeping and multi-trial runner utilities."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from typing import Iterable

import numpy as np

from ddp.model import Job, generate_jobs
from ddp.scripts.run import run_instance

try:  # pragma: no cover - tqdm is optional at runtime
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is absent
    tqdm = None


NUMERIC_TYPES: tuple[type, ...] = (int, float, np.integer, np.floating)


def _parse_values(vals: str) -> list[float]:
    """Parse "1,2,3" or an inclusive range like "1:5:1" (floats allowed)."""

    vals = vals.strip()
    if ":" in vals:
        a, b, *rest = vals.split(":")
        step = float(rest[0]) if rest else 1.0
        a_f, b_f = float(a), float(b)
        if step == 0:
            raise ValueError("step size must be non-zero")
        count = int(round((b_f - a_f) / step)) + 1
        return [a_f + i * step for i in range(count)]
    return [float(x) for x in vals.split(",") if x.strip()]


def _is_nan_number(value: object) -> bool:
    """Return ``True`` when *value* represents a NaN numeric value."""

    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _prepare_jobs(
    n: int,
    rng: np.random.Generator,
    *,
    fix_origin_zero: bool,
    flatten_axis: str | None,
) -> list[Job]:
    """Generate jobs and apply optional geometric transforms."""

    jobs = generate_jobs(n, rng)
    if fix_origin_zero:
        jobs = [Job(origin=(0.0, 0.0), dest=job.dest, timestamp=job.timestamp) for job in jobs]

    if flatten_axis is not None:
        axis = 0 if flatten_axis == "x" else 1

        def _flatten(point: Iterable[float]) -> tuple[float, float]:
            coords = [float(point[0]), float(point[1])]
            coords[axis] = 0.0
            return coords[0], coords[1]

        jobs = [
            Job(origin=_flatten(job.origin), dest=_flatten(job.dest), timestamp=job.timestamp)
            for job in jobs
        ]

    return jobs


def _aggregate_results(
    buckets,
    *,
    n: int,
    d: float,
    trials: int,
    seed0: int,
    extra_meta: dict[str, float | int | str] | None,
) -> tuple[list[dict], list[str]]:
    """Aggregate rows by ``(shadow, dispatch)`` and compute summary statistics."""

    meta_keys = {"n", "d", "seed", "shadow", "dispatch"}
    numeric_keys: set[str] = set()
    for rows in buckets.values():
        for rec in rows:
            for key, value in rec.items():
                if key in meta_keys:
                    continue
                if isinstance(value, NUMERIC_TYPES) and not _is_nan_number(value):
                    numeric_keys.add(key)

    numeric_order = sorted(numeric_keys)

    base_meta: dict[str, float | int | str] = {"n": n, "d": d, "trials": trials, "seed0": seed0}
    if extra_meta:
        base_meta.update(extra_meta)

    aggregated: list[dict] = []
    for (shadow, dispatch), rows in sorted(buckets.items()):
        row = {**base_meta, "shadow": shadow, "dispatch": dispatch}
        for key in numeric_order:
            values: list[float] = []
            for rec in rows:
                raw_val = rec.get(key)
                if isinstance(raw_val, NUMERIC_TYPES) and not _is_nan_number(raw_val):
                    values.append(float(raw_val))
            if not values:
                mean = float("nan")
                std = float("nan")
            elif len(values) == 1:
                mean = values[0]
                std = 0.0
            else:
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1))
            row[f"mean_{key}"] = mean
            row[f"std_{key}"] = std
        aggregated.append(row)

    return aggregated, numeric_order


def _run_trials_for_config(
    *,
    n: int,
    d: float,
    args: argparse.Namespace,
    shadows: list[str],
    dispatches: list[str],
    desc: str | None,
    extra_meta: dict[str, float | int | str] | None,
) -> tuple[list[dict], list[str]]:
    """Run ``args.trials`` experiments for a specific ``(n, d)`` configuration."""

    if n <= 1:
        raise ValueError("Number of jobs n must exceed one")

    buckets = defaultdict(list)

    show_tip = args.trials > 1 and tqdm is None and not getattr(args, "_tqdm_tip_shown", False)
    if show_tip:
        print("(Tip) Install tqdm for a progress bar: pip install tqdm")
        setattr(args, "_tqdm_tip_shown", True)

    pbar = None
    if args.trials > 1 and tqdm is not None:
        pbar = tqdm(total=args.trials, desc=desc or "Trials", unit="trial")

    for offset in range(args.trials):
        seed = args.seed0 + offset
        rng = np.random.default_rng(seed)
        jobs = _prepare_jobs(
            n,
            rng,
            fix_origin_zero=args.fix_origin_zero,
            flatten_axis=args.flatten_axis,
        )
        result = run_instance(
            jobs=jobs,
            d=d,
            shadows=shadows,
            dispatches=dispatches,
            seed=seed,
            with_opt=args.with_opt,
            opt_method=args.opt_method,
            save_csv="",
            print_table=False,
            return_details=False,
            print_matches=False,
        )
        for record in result["rows"]:
            buckets[(record["shadow"], record["dispatch"])].append(record)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return _aggregate_results(
        buckets,
        n=n,
        d=d,
        trials=args.trials,
        seed0=args.seed0,
        extra_meta=extra_meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple trials for SHADOW×DISPATCH configurations, optionally sweeping "
            "a parameter across values."
        )
    )
    parser.add_argument("--param", choices=["d", "n"], help="Parameter to sweep (optional)")
    parser.add_argument(
        "--values",
        help="Comma list '1,2,3' or inclusive range 'start:stop:step' for --param sweeps",
    )
    parser.add_argument("--n", type=int, default=500, help="Number of jobs per trial")
    parser.add_argument("--d", type=float, default=5.0, help="Time window parameter d")
    parser.add_argument("--trials", type=int, default=20, help="Trials per configuration")
    parser.add_argument(
        "--seed0",
        "--seed",
        dest="seed0",
        type=int,
        default=0,
        help="Starting seed; uses seed0..seed0+trials-1",
    )
    parser.add_argument("--shadows", default="naive,pb,hd", help="Comma-separated shadow list")
    parser.add_argument(
        "--dispatch",
        default="greedy,greedy+,batch,rbatch",
        help="Comma-separated dispatch policies",
    )
    parser.add_argument("--outdir", default="results", help="Directory for CSV output")
    parser.add_argument(
        "--save_csv",
        default="results_agg.csv",
        help="Filename for aggregated CSV (written inside --outdir unless absolute)",
    )
    parser.add_argument("--with_opt", action="store_true", help="Compute OPT baseline as well")
    parser.add_argument(
        "--opt_method",
        default="auto",
        choices=["auto", "networkx", "ilp"],
        help="Optimization backend when --with_opt is supplied",
    )
    parser.add_argument(
        "--fix_origin_zero",
        action="store_true",
        help="Set every generated job origin to the depot at (0, 0)",
    )
    parser.add_argument(
        "--flatten_axis",
        choices=["x", "y"],
        help="Project all jobs onto a single axis by zeroing the chosen coordinate",
    )

    args = parser.parse_args()

    sweep_mode = args.param is not None or args.values is not None
    if sweep_mode and not (args.param and args.values):
        parser.error("Both --param and --values must be provided to sweep a parameter")
    if not sweep_mode and args.values:
        parser.error("--values requires --param to be specified")

    shadows = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    if not shadows:
        parser.error("No valid shadows supplied via --shadows")
    dispatches = [d.strip().lower() for d in args.dispatch.split(",") if d.strip()]
    if not dispatches:
        parser.error("No valid dispatch policies supplied via --dispatch")

    if args.save_csv:
        if os.path.isabs(args.save_csv) or not args.outdir:
            save_path = args.save_csv
        else:
            os.makedirs(args.outdir, exist_ok=True)
            save_path = os.path.join(args.outdir, args.save_csv)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    else:
        save_path = ""

    t0 = time.perf_counter()

    all_rows: list[dict] = []
    metric_union: set[str] = set()

    if sweep_mode:
        values = _parse_values(args.values)
        if args.param == "d":
            print(
                f"Sweeping d over {values} | trials={args.trials} | n={args.n} (fixed)",
                flush=True,
            )
        else:
            print(
                f"Sweeping n over {values} | trials={args.trials} | d={args.d} (fixed)",
                flush=True,
            )

        for value in values:
            if args.param == "d":
                current_n = args.n
                current_d = float(value)
            else:
                current_n = int(round(value))
                current_d = args.d
            print(
                f"\n== Sweeping {args.param} = {value} (n={current_n}, d={current_d}) | "
                f"trials={args.trials} ==",
                flush=True,
            )
            rows, metrics = _run_trials_for_config(
                n=current_n,
                d=current_d,
                args=args,
                shadows=shadows,
                dispatches=dispatches,
                desc=f"{args.param}={value}",
                extra_meta={"param": args.param, "param_value": float(value)},
            )
            all_rows.extend(rows)
            metric_union.update(metrics)
    else:
        rows, metrics = _run_trials_for_config(
            n=args.n,
            d=args.d,
            args=args,
            shadows=shadows,
            dispatches=dispatches,
            desc=f"n={args.n}, d={args.d}",
            extra_meta=None,
        )
        all_rows.extend(rows)
        metric_union.update(metrics)

    metric_list = sorted(metric_union)
    for row in all_rows:
        for metric in metric_list:
            row.setdefault(f"mean_{metric}", float("nan"))
            row.setdefault(f"std_{metric}", float("nan"))

    duration = time.perf_counter() - t0

    if all_rows and save_path:
        fieldnames = []
        if any("param" in row for row in all_rows):
            fieldnames.extend(["param", "param_value"])
        fieldnames.extend(["shadow", "dispatch", "n", "d", "trials", "seed0"])
        fieldnames.extend([f"mean_{m}" for m in metric_list])
        fieldnames.extend([f"std_{m}" for m in metric_list])
        with open(save_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        abs_path = os.path.abspath(save_path)
        print(f"Wrote {abs_path} with {len(all_rows)} rows in {duration:.2f}s")
    else:
        print(f"Completed {len(all_rows)} aggregated rows in {duration:.2f}s")

    if metric_list:
        print("Aggregated metrics:", ", ".join(metric_list))
    else:
        print("Aggregated metrics: (none)")


if __name__ == "__main__":
    main()

