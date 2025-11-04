"""Run gamma/tau sweeps on Meituan CSV snapshots without geometry transforms."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

from ddp.model import Job
from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import (
    AverageDualError,
    AverageDualTable,
    load_average_dual_mapper,
    load_average_duals,
)
from ddp.scripts.shadow_sweep import (
    ALLOWED_METRICS,
    ALL_SHADOWS,
    DEFAULT_SHADOWS,
    GeometryPreset,
    _identity,
    _parse_name_list,
    _parse_values,
    _plot_heatmaps,
    _run_sweep_from_trial_jobs,
    _write_csv,
)


_DATASET_GEOMETRY = GeometryPreset(
    name="dataset",
    label="Raw dataset",
    transform=_identity,
)


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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the Meituan CSV sweep CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.limit < 0:
        parser.error("--limit must be non-negative")

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
        heatmap, records, best_entries = _run_sweep_from_trial_jobs(
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

