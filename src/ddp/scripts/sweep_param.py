"""Unified parameter sweeping and multi-trial runner utilities."""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ddp.model import Job, generate_jobs
from ddp.scripts.run import (
    AverageDualTable,
    load_average_dual_mapper,
    load_average_duals,
    run_instance,
)

try:  # pragma: no cover - tqdm is optional at runtime
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is absent
    tqdm = None


CSV_FIELD_ORDER = [
    "param",
    "param_value",
    "trial_index",
    "seed",
    "n_fixed",
    "d_fixed",
    "shadow",
    "dispatch",
    "n",
    "d",
    "savings",
    "pooled_pct",
    "ratio_lp",
    "lp_gap",
    "ratio_opt",
    "opt_gap",
    "pairs",
    "solos",
    "time_s",
    "method",
]


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


def _run_trials_for_config(
    *,
    n: int,
    d: float,
    args: argparse.Namespace,
    shadows: list[str],
    dispatches: list[str],
    desc: str | None,
    extra_meta: dict[str, float | int | str | None] | None,
    ad_duals: AverageDualTable
    | Mapping[object, float]
    | Sequence[float]
    | np.ndarray
    | None,
    ad_mapper: Callable[[Job], str | None] | None,
) -> Iterator[dict]:
    """Yield rows for ``args.trials`` experiments at a given ``(n, d)`` setting."""

    if n <= 1:
        raise ValueError("Number of jobs n must exceed one")

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
            ad_duals=ad_duals,
            ad_mapper=ad_mapper,
        )
        for record in result["rows"]:
            enriched = dict(record)
            enriched["seed"] = seed
            enriched["trial_index"] = offset
            enriched["n"] = n
            enriched["d"] = d
            if extra_meta:
                enriched.update(extra_meta)
            yield enriched
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()


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
    parser.add_argument("--n", type=int, default=100, help="Number of jobs per trial")
    parser.add_argument("--d", type=float, default=2.0, help="Time window parameter d")
    parser.add_argument("--trials", type=int, default=10, help="Trials per configuration")
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
        default="greedy,greedy+,batch,batch+,rbatch,rbatch+",
        help=(
            "Comma-separated dispatch policies. Include 'batch+'/'rbatch+' to evaluate "
            "the late-arrival shadow variants."
        ),
    )
    parser.add_argument("--outdir", default="results", help="Directory for CSV output")
    parser.add_argument(
        "--save_csv",
        default="results_trials.csv",
        help=(
            "Filename for per-trial CSV (written inside --outdir unless absolute); "
            "rows are streamed as trials finish"
        ),
    )
    parser.add_argument(
        "--with_opt",
        action="store_true",
        help="Compute the OPT baseline respecting the deadline parameter 'd'",
    )
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
    parser.add_argument(
        "--ad_duals",
        help=(
            "Path to an average-dual table (.npz or CSV). Required when --shadows includes 'ad'."
        ),
    )
    parser.add_argument(
        "--ad-mapping",
        help=(
            "Module:function resolving to an average-dual mapper for type-indexed tables."
        ),
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

    ad_table: AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray | None = (
        load_average_duals(args.ad_duals) if args.ad_duals else None
    )
    ad_mapper: Callable[[Job], str | None] | None = None
    if "ad" in shadows:
        if ad_table is None:
            parser.error("--shadows includes 'ad' so --ad_duals must be provided")
        if args.ad_mapping:
            try:
                ad_mapper = load_average_dual_mapper(args.ad_mapping)
            except (ModuleNotFoundError, AttributeError, ValueError, TypeError) as exc:
                parser.error(f"failed to resolve --ad-mapping: {exc}")
        if (
            isinstance(ad_table, AverageDualTable)
            and ad_table.by_job is None
            and ad_table.by_type is not None
            and ad_mapper is None
        ):
            parser.error("type-indexed average-dual tables require --ad-mapping")
    elif args.ad_mapping:
        parser.error("--ad-mapping can only be used when --shadows includes 'ad'")

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

    total_rows = 0
    csv_handle = None
    writer: csv.DictWriter | None = None
    fieldnames: list[str] | None = None
    t0 = time.perf_counter()

    try:
        def record_row(row: dict) -> None:
            nonlocal total_rows, writer, csv_handle, fieldnames
            total_rows += 1
            if not save_path:
                return
            if writer is None:
                ordered = [field for field in CSV_FIELD_ORDER if field in row]
                extras = [key for key in row if key not in ordered]
                fieldnames = ordered + sorted(extras)
                csv_handle = open(save_path, "w", newline="")
                writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
                writer.writeheader()
            assert writer is not None and csv_handle is not None
            writer.writerow(row)
            csv_handle.flush()

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
                extra_meta = {
                    "param": args.param,
                    "param_value": float(value),
                    "n_fixed": args.n if args.param == "d" else None,
                    "d_fixed": args.d if args.param == "n" else None,
                }
                for row in _run_trials_for_config(
                    n=current_n,
                    d=current_d,
                    args=args,
                    shadows=shadows,
                    dispatches=dispatches,
                    desc=f"{args.param}={value}",
                    extra_meta=extra_meta,
                    ad_duals=ad_table,
                    ad_mapper=ad_mapper,
                ):
                    record_row(row)
        else:
            extra_meta = {
                "param": None,
                "param_value": None,
                "n_fixed": args.n,
                "d_fixed": args.d,
            }
            for row in _run_trials_for_config(
                n=args.n,
                d=args.d,
                args=args,
                shadows=shadows,
                dispatches=dispatches,
                desc=f"n={args.n}, d={args.d}",
                extra_meta=extra_meta,
                ad_duals=ad_table,
                ad_mapper=ad_mapper,
            ):
                record_row(row)
    finally:
        if csv_handle is not None:
            csv_handle.close()

    duration = time.perf_counter() - t0

    if save_path:
        abs_path = os.path.abspath(save_path)
        if writer is not None:
            print(f"Wrote {abs_path} with {total_rows} rows in {duration:.2f}s")
        else:
            print(f"No rows recorded; nothing written to {abs_path} ({duration:.2f}s)")
    else:
        print(f"Completed {total_rows} trial rows in {duration:.2f}s")


if __name__ == "__main__":
    main()

