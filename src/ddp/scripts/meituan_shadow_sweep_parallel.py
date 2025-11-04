"""Run gamma/tau sweeps on Meituan CSV snapshots using process-level parallelism."""

from __future__ import annotations

import argparse
import glob
import math
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ddp.model import Job
from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import (
    AverageDualError,
    AverageDualTable,
    load_average_dual_mapper,
    load_average_duals,
    run_instance,
)
from ddp.scripts.shadow_sweep import (
    ALLOWED_METRICS,
    ALL_SHADOWS,
    DEFAULT_SHADOWS,
    GeometryPreset,
    _extract_metric,
    _identity,
    _parse_name_list,
    _parse_values,
    _plot_heatmaps,
    _write_csv,
)


_DATASET_GEOMETRY = GeometryPreset(
    name="dataset",
    label="Raw dataset",
    transform=_identity,
)


@dataclass(frozen=True)
class ParallelSweepTask:
    """Describe a single shadow sweep evaluation task."""

    geometry_name: str
    shadow: str
    tau_index: int
    gamma_index: int
    tau: float
    gamma: float
    seed: int
    jobs: Sequence[Job]
    ad_duals_override: AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray | None
    tau_s: float


def _expand_jobs_paths(patterns: Iterable[str]) -> list[Path]:
    """Expand job CSV patterns into concrete paths."""

    resolved: list[Path] = []
    seen: set[Path] = set()

    for pattern in patterns:
        matches = [Path(match) for match in sorted(glob.glob(pattern, recursive=True))]
        if not matches:
            matches = [Path(pattern)]

        for path in matches:
            if not path.exists():
                msg = f"jobs CSV not found: {path}"
                raise FileNotFoundError(msg)
            if path in seen:
                continue
            resolved.append(path)
            seen.add(path)

    return resolved


def _execute_task(
    task: ParallelSweepTask,
    *,
    d: float,
    dispatch: str,
    metric: str,
    ad_mapper: Callable[[Job], str | None] | None,
) -> tuple[str, str, int, int, float | None]:
    """Worker entry point for evaluating a single sweep configuration."""

    result = run_instance(
        jobs=list(task.jobs),
        d=d,
        shadows=[task.shadow],
        dispatches=[dispatch],
        seed=task.seed,
        with_opt=False,
        save_csv="",
        print_table=False,
        return_details=False,
        print_matches=False,
        gamma=float(task.gamma),
        tau=float(task.tau),
        tau_s=task.tau_s,
        ad_duals=task.ad_duals_override,
        ad_mapper=ad_mapper,
    )
    row = result["rows"][0]
    metric_value = _extract_metric(row, metric)
    return (
        task.geometry_name,
        task.shadow,
        task.tau_index,
        task.gamma_index,
        metric_value,
    )


def _run_parallel_sweep_from_trial_jobs(
    *,
    d: float,
    dispatch: str,
    metric: str,
    gamma_values: Sequence[float],
    tau_values: Sequence[float],
    geometries: Sequence[GeometryPreset],
    shadows: Sequence[str],
    ad_duals: AverageDualTable
    | Mapping[object, float]
    | Sequence[float]
    | np.ndarray
    | None,
    ad_mapper: Callable[[Job], str | None] | None,
    trial_jobs: Sequence[tuple[int, Mapping[str, Sequence[Job]]]],
    trial_ad_duals: Mapping[
        int,
        AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray,
    ]
    | Sequence[AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray]
    | None = None,
    tau_s: float = 30.0,
    progress: bool = True,
    workers: int | None = None,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    list[dict],
    list[tuple[GeometryPreset, str, float | None, float | None, float | None]],
]:
    """Execute the gamma/tau sweep using a process pool."""

    trial_jobs = list(trial_jobs)

    heatmap: dict[str, dict[str, np.ndarray]] = {
        geom.name: {
            shadow: np.full((len(tau_values), len(gamma_values)), np.nan, dtype=float)
            for shadow in shadows
        }
        for geom in geometries
    }

    records: list[dict] = []
    best_entries: list[tuple[GeometryPreset, str, float | None, float | None, float | None]] = []

    trials = len(trial_jobs)
    tasks: list[ParallelSweepTask] = []

    for tau_index, tau in enumerate(tau_values):
        for gamma_index, gamma in enumerate(gamma_values):
            for geom in geometries:
                for shadow in shadows:
                    for trial_index, (seed, job_variants) in enumerate(trial_jobs):
                        jobs = job_variants[geom.name]
                        ad_duals_override = ad_duals
                        if trial_ad_duals is not None:
                            if isinstance(trial_ad_duals, Mapping):
                                ad_duals_override = trial_ad_duals.get(seed, ad_duals_override)
                            else:
                                if 0 <= trial_index < len(trial_ad_duals):
                                    ad_duals_override = trial_ad_duals[trial_index]
                        tasks.append(
                            ParallelSweepTask(
                                geometry_name=geom.name,
                                shadow=shadow,
                                tau_index=tau_index,
                                gamma_index=gamma_index,
                                tau=float(tau),
                                gamma=float(gamma),
                                seed=seed,
                                jobs=jobs,
                                ad_duals_override=ad_duals_override,
                                tau_s=float(tau_s),
                            )
                        )

    total_trials = len(tasks)
    show_progress = progress and total_trials > 0
    progress_bar = tqdm(
        total=total_trials,
        desc="Trials",
        unit="trial",
        disable=not show_progress,
    )

    values: defaultdict[tuple[str, str, int, int], list[float]] = defaultdict(list)

    max_workers = workers if workers is not None else None

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _execute_task,
                    task,
                    d=d,
                    dispatch=dispatch,
                    metric=metric,
                    ad_mapper=ad_mapper,
                ): task
                for task in tasks
            }

            for future in as_completed(futures):
                geometry_name, shadow, tau_index, gamma_index, metric_value = future.result()
                if metric_value is not None:
                    values[(geometry_name, shadow, tau_index, gamma_index)].append(metric_value)
                progress_bar.update()
    finally:
        progress_bar.close()

    for geom in geometries:
        for shadow in shadows:
            best_mean: float | None = None
            best_gamma: float | None = None
            best_tau: float | None = None

            for tau_index, tau in enumerate(tau_values):
                for gamma_index, gamma in enumerate(gamma_values):
                    key = (geom.name, shadow, tau_index, gamma_index)
                    metric_values = values.get(key, [])

                    if metric_values:
                        mean_val = statistics.fmean(metric_values)
                        std_val = (
                            statistics.stdev(metric_values)
                            if len(metric_values) > 1
                            else 0.0
                        )
                        heatmap[geom.name][shadow][tau_index, gamma_index] = mean_val
                        if best_mean is None or mean_val > best_mean:
                            best_mean = mean_val
                            best_gamma = float(gamma)
                            best_tau = float(tau)
                    else:
                        mean_val = math.nan
                        std_val = math.nan

                    records.append(
                        {
                            "geometry": geom.name,
                            "shadow": shadow,
                            "gamma": float(gamma),
                            "tau": float(tau),
                            "tau_s": float(tau_s),
                            "dispatch": dispatch,
                            "metric": metric,
                            "mean": mean_val,
                            "std": std_val,
                            "trials": trials,
                            "valid_trials": len(metric_values),
                        }
                    )

            best_entries.append((geom, shadow, best_mean, best_gamma, best_tau))

    return heatmap, records, best_entries


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for Meituan CSV sweeps."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-csv",
        action="append",
        required=True,
        dest="jobs_csv",
        help=(
            "Path or glob pattern for a Meituan-style CSV. Can be provided multiple "
            "times; each file becomes a separate trial."
        ),
    )
    parser.add_argument(
        "--timestamp-column",
        default="platform_order_time",
        help="Column containing ISO 8601 timestamps (default: platform_order_time)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on the number of jobs loaded from each CSV (0 = no limit)",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=20.0,
        help="Time window parameter d shared by every trial (seconds)",
    )
    parser.add_argument(
        "--dispatch",
        choices=["greedy", "greedy+", "batch", "batch+", "rbatch", "rbatch+", "batch2", "rbatch2"],
        default="greedy",
        help="Dispatch policy to evaluate",
    )
    parser.add_argument(
        "--metric",
        choices=ALLOWED_METRICS,
        default="savings",
        help="Metric aggregated across trials",
    )
    parser.add_argument(
        "--gamma-values",
        dest="gamma_values",
        default="1.0",
        help="Comma list or start:stop:step grid for gamma scaling values",
    )
    parser.add_argument(
        "--tau-values",
        dest="tau_values",
        default="0.0",
        help="Comma list or start:stop:step grid for tau offset values",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to save the heatmap figure (PNG, PDF, etc.)",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Optional path to save aggregated metric summaries as CSV",
    )
    parser.add_argument(
        "--shadows",
        default=",".join(DEFAULT_SHADOWS),
        help="Comma list of shadow potentials to evaluate (choices: %s)"
        % ", ".join(ALL_SHADOWS),
    )
    parser.add_argument(
        "--ad-duals",
        default="",
        help=(
            "Path to a precomputed average-dual table (.npz or CSV). Required when "
            "--shadows includes 'ad'."
        ),
    )
    parser.add_argument(
        "--ad-mapping",
        default="",
        help=(
            "Module:function resolving to an average-dual mapper for type-indexed tables."
        ),
    )
    parser.add_argument(
        "--tau_s",
        type=float,
        default=30.0,
        help="Period (seconds) between matching evaluations for batch2/rbatch2.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the generated heatmaps interactively",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes to launch (defaults to CPU count). Use 1 to "
            "disable parallelism."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the Meituan CSV sweep CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be a positive integer")

    try:
        gamma_values = _parse_values(args.gamma_values)
    except ValueError as exc:
        parser.error(f"invalid --gamma-values: {exc}")
    try:
        tau_values = _parse_values(args.tau_values)
    except ValueError as exc:
        parser.error(f"invalid --tau-values: {exc}")

    try:
        shadow_names = _parse_name_list(args.shadows)
    except ValueError as exc:
        parser.error(f"invalid --shadows: {exc}")

    shadows: list[str] = []
    seen_shadows: set[str] = set()
    for name in shadow_names:
        if name not in ALL_SHADOWS:
            parser.error(f"unknown shadow '{name}' (choices: {', '.join(ALL_SHADOWS)})")
        if name in seen_shadows:
            continue
        shadows.append(name)
        seen_shadows.add(name)
    if not shadows:
        parser.error("at least one shadow family must be selected")

    try:
        jobs_paths = _expand_jobs_paths(args.jobs_csv)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not jobs_paths:
        parser.error("no CSV files matched --jobs-csv")

    ad_duals: AverageDualTable | Mapping[object, float] | Sequence[float] | None = None
    ad_mapper: Callable[[Job], str | None] | None = None
    if "ad" in shadows:
        if not args.ad_duals:
            parser.error("--ad-duals is required when evaluating AD shadows")
        try:
            ad_duals = load_average_duals(args.ad_duals)
        except (OSError, ValueError) as exc:
            parser.error(f"failed to load --ad-duals: {exc}")
        if args.ad_mapping:
            try:
                ad_mapper = load_average_dual_mapper(args.ad_mapping)
            except (ModuleNotFoundError, AttributeError, ValueError, TypeError) as exc:
                parser.error(f"failed to resolve --ad-mapping: {exc}")
        if (
            isinstance(ad_duals, AverageDualTable)
            and ad_duals.by_job is None
            and ad_duals.by_type is not None
            and ad_mapper is None
        ):
            parser.error("type-indexed average-dual tables require --ad-mapping")
    elif args.ad_mapping:
        parser.error("--ad-mapping can only be used when --shadows includes 'ad'")

    trial_jobs: list[tuple[int, dict[str, Sequence[Job]]]] = []
    for index, csv_path in enumerate(jobs_paths):
        jobs = load_jobs_from_csv(csv_path, timestamp_column=args.timestamp_column)
        if args.limit:
            jobs = jobs[: args.limit]
        if not jobs:
            parser.error(f"no valid jobs found in {csv_path}")
        trial_jobs.append((index, {_DATASET_GEOMETRY.name: jobs}))

    try:
        heatmap, records, best_entries = _run_parallel_sweep_from_trial_jobs(
            d=float(args.d),
            dispatch=args.dispatch,
            metric=args.metric,
            gamma_values=gamma_values,
            tau_values=tau_values,
            geometries=[_DATASET_GEOMETRY],
            shadows=shadows,
            ad_duals=ad_duals,
            ad_mapper=ad_mapper,
            trial_jobs=trial_jobs,
            tau_s=args.tau_s,
            progress=not args.no_progress,
            workers=args.workers,
        )
    except AverageDualError as exc:
        parser.error(str(exc))

    print("Loaded trials:")
    for csv_path, trial in zip(jobs_paths, trial_jobs):
        print(f"  {csv_path}: {len(trial[1][_DATASET_GEOMETRY.name])} jobs")

    print("\nBest configurations:")
    for geom, shadow, best_mean, best_gamma, best_tau in best_entries:
        label = f"[{geom.label}] {shadow.upper()}"
        if best_gamma is None or best_tau is None or best_mean is None:
            print(f"  {label}: no valid {args.metric} values")
        else:
            print(
                f"  {label}: mean {args.metric}={best_mean:.3f} at gamma={best_gamma:g}, tau={best_tau:g}"
            )

    fig = _plot_heatmaps(
        heatmap,
        gamma_values,
        tau_values,
        args.metric,
        [_DATASET_GEOMETRY],
        shadows,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    _write_csv(records, args.csv)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

