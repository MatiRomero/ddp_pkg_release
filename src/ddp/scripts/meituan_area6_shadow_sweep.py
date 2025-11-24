"""Run gamma/tau sweeps on the eight-day Meituan Area 6 dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ddp.model import Job
from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import load_average_dual_mapper, load_average_duals
from ddp.scripts.shadow_sweep import (
    ALLOWED_METRICS,
    GeometryPreset,
    _identity,
    _parse_name_list,
    _parse_values,
    _run_sweep_from_trial_jobs,
)


_DATASET_GEOMETRY = GeometryPreset(
    name="dataset",
    label="Raw dataset",
    transform=_identity,
)


@dataclass(frozen=True)
class _DayConfig:
    day: int
    jobs: Sequence[Job]
    ad_duals_path: Path | None


def _format_parameter(value: float) -> str:
    """Format float parameters to match lookup filenames."""

    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _load_day_jobs(day: int, *, timestamp_column: str = "platform_order_time") -> Sequence[Job]:
    csv_path = Path(f"data/meituan_area6_lunchtime_plat10301330_day{day}.csv")
    if not csv_path.exists():
        msg = f"jobs CSV not found for day {day}: {csv_path}"
        raise FileNotFoundError(msg)
    jobs = load_jobs_from_csv(csv_path, timestamp_column=timestamp_column)
    if not jobs:
        msg = f"no valid jobs loaded from {csv_path}"
        raise ValueError(msg)
    return jobs


def _build_day_configs(*, include_ad: bool, d: float, resolution: int) -> list[_DayConfig]:
    formatted_d = _format_parameter(d)
    formatted_res = str(int(resolution))
    configs: list[_DayConfig] = []

    for day in range(8):
        ad_duals_path: Path | None = None
        if include_ad:
            ad_duals_path = Path(
                "data/average_duals/"
                f"meituan_area6_ad_day{day}_d{formatted_d}_res{formatted_res}_lookup.csv"
            )
            if not ad_duals_path.exists():
                msg = f"AD lookup missing for day {day}: {ad_duals_path}"
                raise FileNotFoundError(msg)
        jobs = _load_day_jobs(day)
        configs.append(_DayConfig(day=day, jobs=jobs, ad_duals_path=ad_duals_path))

    return configs


def _write_summary_csv(rows: Iterable[Mapping[str, object]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "d",
        "dispatch",
        "shadow",
        "gamma",
        "tau",
        "tau_s",
        "resolution",
        "metric_mean",
        "metric_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=float, required=True, help="Time window parameter d (seconds)")
    parser.add_argument(
        "--dispatch",
        choices=["greedy", "greedyx", "greedy+", "batch", "batch+", "rbatch", "rbatch+", "batch2", "rbatch2"],
        required=True,
        help="Dispatch policy to evaluate",
    )
    parser.add_argument(
        "--metric",
        choices=ALLOWED_METRICS,
        default="savings",
        help="Metric aggregated across the eight days",
    )
    parser.add_argument(
        "--gamma-values",
        dest="gamma_values",
        required=True,
        help="Comma list or start:stop:step grid for gamma scaling values",
    )
    parser.add_argument(
        "--tau-values",
        dest="tau_values",
        required=True,
        help="Comma list or start:stop:step grid for tau offset values",
    )
    parser.add_argument(
        "--shadows",
        default="naive,pb,hd",
        help="Comma-separated list of shadow families to evaluate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help="H3 resolution used for AD lookups",
    )
    parser.add_argument(
        "--csv",
        default="results/meituan_area6_shadow_summary.csv",
        help="Path where the summary CSV will be written",
    )
    parser.add_argument(
        "--tau_s",
        type=float,
        default=30.0,
        help="Period (seconds) between matching evaluations for batch2/rbatch2.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

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
    seen: set[str] = set()
    for name in shadow_names:
        if name not in {"naive", "pb", "hd", "ad"}:
            parser.error(f"unknown shadow '{name}' (choices: naive,pb,hd,ad)")
        if name in seen:
            continue
        shadows.append(name)
        seen.add(name)
    if not shadows:
        parser.error("at least one shadow family must be selected")

    include_ad = "ad" in shadows

    try:
        day_configs = _build_day_configs(include_ad=include_ad, d=float(args.d), resolution=args.resolution)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    ad_mapper = None
    if include_ad:
        try:
            ad_mapper = load_average_dual_mapper("ddp.mappings.h3_pairs:job_mapping")
        except (ModuleNotFoundError, AttributeError, ValueError, TypeError) as exc:
            parser.error(f"failed to resolve AD mapping: {exc}")

    trial_jobs: list[tuple[int, dict[str, Sequence[Job]]]] = []
    trial_ad_duals: list[object] = []

    for config in day_configs:
        trial_jobs.append((config.day, {_DATASET_GEOMETRY.name: list(config.jobs)}))
        if include_ad:
            try:
                trial_ad_duals.append(load_average_duals(str(config.ad_duals_path)))
            except (OSError, ValueError) as exc:
                parser.error(f"failed to load AD lookup for day {config.day}: {exc}")

    _, records, _ = _run_sweep_from_trial_jobs(
        d=float(args.d),
        dispatch=args.dispatch,
        metric=args.metric,
        gamma_values=gamma_values,
        tau_values=tau_values,
        geometries=[_DATASET_GEOMETRY],
        shadows=shadows,
        ad_duals=None,
        ad_mapper=ad_mapper,
        trial_jobs=trial_jobs,
        trial_ad_duals=trial_ad_duals if include_ad else None,
        tau_s=args.tau_s,
        progress=not args.no_progress,
    )

    summary_rows = []
    for record in records:
        if record["geometry"] != _DATASET_GEOMETRY.name:
            continue
        summary_rows.append(
            {
                "d": float(args.d),
                "dispatch": args.dispatch,
                "shadow": record["shadow"],
                "gamma": record["gamma"],
                "tau": record["tau"],
                "tau_s": args.tau_s,
                "resolution": args.resolution,
                "metric_mean": record["mean"],
                "metric_std": record["std"],
            }
        )

    output_path = Path(args.csv)
    _write_summary_csv(summary_rows, output_path)

    print(f"Summary written to {output_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

