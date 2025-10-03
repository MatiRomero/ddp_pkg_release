"""Gamma/Tau sweeps for SHADOW×DISPATCH configurations across geometries."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ddp.model import Job, generate_jobs
from ddp.scripts.run import (
    AverageDualError,
    AverageDualTable,
    load_average_dual_mapper,
    load_average_duals,
    run_instance,
)


DEFAULT_SHADOWS: tuple[str, ...] = ("naive", "pb", "hd")
ALL_SHADOWS: tuple[str, ...] = DEFAULT_SHADOWS + ("ad",)


@dataclass(frozen=True)
class GeometryPreset:
    """Description of a geometry transformation applied to generated jobs."""

    name: str
    label: str
    transform: Callable[[Sequence[Job]], list[Job]]


def _flatten_to_line(jobs: Sequence[Job]) -> list[Job]:
    """Project origins and destinations onto the x-axis (y = 0)."""

    flattened: list[Job] = []
    for job in jobs:
        origin = (float(job.origin[0]), 0.0)
        dest = (float(job.dest[0]), 0.0)
        flattened.append(Job(origin=origin, dest=dest, timestamp=job.timestamp))
    return flattened


def _pin_origins(jobs: Sequence[Job]) -> list[Job]:
    """Place every job origin at the depot while keeping destinations intact."""

    return [
        Job(origin=(0.0, 0.0), dest=job.dest, timestamp=job.timestamp)
        for job in jobs
    ]


def _identity(jobs: Sequence[Job]) -> list[Job]:
    """Return a shallow copy of the job list (baseline 2-D geometry)."""

    return list(jobs)


GEOMETRY_PRESETS: dict[str, GeometryPreset] = {
    preset.name: preset
    for preset in (
        GeometryPreset(
            name="line_y0",
            label="1D line (y=0)",
            transform=_flatten_to_line,
        ),
        GeometryPreset(
            name="common_origin",
            label="Common origin",
            transform=_pin_origins,
        ),
        GeometryPreset(
            name="plane",
            label="Random 2D",
            transform=_identity,
        ),
    )
}
DEFAULT_GEOMETRIES: tuple[str, ...] = ("plane",)


ALLOWED_METRICS: tuple[str, ...] = (
    "savings",
    "pooled_pct",
    "ratio_lp",
    "lp_gap",
    "ratio_opt",
    "opt_gap",
    "pairs",
    "solos",
    "time_s",
)


@dataclass(frozen=True)
class SweepCombinationResult:
    """Aggregate statistics for a single geometry/shadow/gamma/tau combination."""

    mean: float
    std: float
    trials: int
    valid_trials: int


def evaluate_sweep_combination(
    *,
    geometry: str,
    shadow: str,
    gamma: float,
    tau: float,
    d: float,
    dispatch: str,
    metric: str,
    trial_jobs: Sequence[tuple[int, Sequence[Job]]],
    ad_duals: AverageDualTable
    | Mapping[object, float]
    | Sequence[float]
    | np.ndarray
    | None,
    ad_mapper: Callable[[Job], str | None] | None,
    trial_ad_duals: Mapping[
        int,
        AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray,
    ]
    | Sequence[
        AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray
    ]
    | None = None,
    progress: Callable[[int], None] | None = None,
) -> SweepCombinationResult:
    """Run all trials for a specific combination and aggregate the metric."""

    values: list[float] = []
    for trial_index, (seed, jobs) in enumerate(trial_jobs):
        ad_duals_override = ad_duals
        if trial_ad_duals is not None:
            if isinstance(trial_ad_duals, Mapping):
                ad_duals_override = trial_ad_duals.get(seed, ad_duals_override)
            else:
                if 0 <= trial_index < len(trial_ad_duals):
                    ad_duals_override = trial_ad_duals[trial_index]
        result = run_instance(
            jobs=jobs,
            d=d,
            shadows=[shadow],
            dispatches=[dispatch],
            seed=seed,
            with_opt=False,
            save_csv="",
            print_table=False,
            return_details=False,
            print_matches=False,
            gamma=float(gamma),
            tau=float(tau),
            ad_duals=ad_duals_override,
            ad_mapper=ad_mapper,
        )
        row = result["rows"][0]
        metric_value = _extract_metric(row, metric)
        if metric_value is not None:
            values.append(metric_value)
        if progress is not None:
            progress(1)

    trials = len(trial_jobs)
    if values:
        mean_val = statistics.fmean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
    else:
        mean_val = float("nan")
        std_val = float("nan")

    return SweepCombinationResult(
        mean=mean_val,
        std=std_val,
        trials=trials,
        valid_trials=len(values),
    )


def _parse_values(spec: str) -> list[float]:
    """Parse comma lists or inclusive ``start:stop:step`` ranges into floats."""

    spec = spec.strip()
    if not spec:
        raise ValueError("value specification must be non-empty")

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError("range format must be start:stop[:step]")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 1.0
        if step == 0:
            raise ValueError("step size must be non-zero")
        count = int(round((stop - start) / step)) + 1
        if count <= 0:
            raise ValueError("range does not produce any values")
        return [start + i * step for i in range(count)]

    values = [float(chunk) for chunk in spec.split(",") if chunk.strip()]
    if not values:
        raise ValueError("no numeric values parsed from specification")
    return values


def _parse_name_list(spec: str) -> list[str]:
    """Parse a comma-delimited list into stripped tokens."""

    values = [chunk.strip() for chunk in spec.split(",") if chunk.strip()]
    if not values:
        raise ValueError("value specification must be non-empty")
    return values


def _prepare_trials(
    n: int,
    trials: int,
    seed0: int,
    geometries: Sequence[GeometryPreset],
) -> list[tuple[int, dict[str, list[Job]]]]:
    """Generate base jobs and geometry variants for each trial seed."""

    prepared: list[tuple[int, dict[str, list[Job]]]] = []
    for offset in range(trials):
        seed = seed0 + offset
        rng = np.random.default_rng(seed)
        base_jobs = generate_jobs(n, rng)
        variants = {preset.name: preset.transform(base_jobs) for preset in geometries}
        prepared.append((seed, variants))
    return prepared


def _extract_metric(row: dict, metric: str) -> float | None:
    """Convert a metric value from ``run_instance`` output into a float."""

    value = row.get(metric)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _format_value(val: float) -> str:
    """Pretty-print numeric grid labels for axes."""

    return f"{val:g}"


def _run_sweep_from_trial_jobs(
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
    trial_ad_duals: Mapping[int, AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray]
    | Sequence[AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray]
    | None = None,
    progress: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict], list[tuple[GeometryPreset, str, float | None, float | None, float | None]]]:
    """Execute the gamma/tau sweep and return heatmap data, records, and best combos."""

    trial_jobs = list(trial_jobs)
    geometry_trial_jobs: dict[str, list[tuple[int, Sequence[Job]]]] = {
        geom.name: [(seed, job_variants[geom.name]) for seed, job_variants in trial_jobs]
        for geom in geometries
    }

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
    total_trials = (
        len(geometries)
        * len(shadows)
        * len(tau_values)
        * len(gamma_values)
        * len(trial_jobs)
    )
    show_progress = progress and total_trials > 0
    progress_bar = tqdm(
        total=total_trials,
        desc="Trials",
        unit="trial",
        disable=not show_progress,
    )

    try:
        for geom in geometries:
            for shadow in shadows:
                best_mean: float | None = None
                best_gamma: float | None = None
                best_tau: float | None = None

                for tau_index, tau in enumerate(tau_values):
                    for gamma_index, gamma in enumerate(gamma_values):
                        if show_progress:
                            progress_bar.set_description(
                                f"{geom.name}/{shadow} γ={_format_value(float(gamma))} τ={_format_value(float(tau))}"
                            )
                        combo_result = evaluate_sweep_combination(
                            geometry=geom.name,
                            shadow=shadow,
                            gamma=float(gamma),
                            tau=float(tau),
                            d=d,
                            dispatch=dispatch,
                            metric=metric,
                            trial_jobs=geometry_trial_jobs[geom.name],
                            ad_duals=ad_duals,
                            ad_mapper=ad_mapper,
                            trial_ad_duals=trial_ad_duals,
                            progress=progress_bar.update if show_progress else None,
                        )

                        heatmap[geom.name][shadow][tau_index, gamma_index] = combo_result.mean
                        if (
                            combo_result.valid_trials > 0
                            and (best_mean is None or combo_result.mean > best_mean)
                        ):
                            best_mean = combo_result.mean
                            best_gamma = float(gamma)
                            best_tau = float(tau)

                        records.append(
                            {
                                "geometry": geom.name,
                                "shadow": shadow,
                                "gamma": float(gamma),
                                "tau": float(tau),
                                "dispatch": dispatch,
                                "metric": metric,
                                "mean": combo_result.mean,
                                "std": combo_result.std,
                                "trials": trials,
                                "valid_trials": combo_result.valid_trials,
                            }
                        )

                best_entries.append((geom, shadow, best_mean, best_gamma, best_tau))
    finally:
        progress_bar.close()

    return heatmap, records, best_entries


def _run_sweep(
    *,
    n: int,
    d: float,
    trials: int,
    seed0: int,
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
    progress: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict], list[tuple[GeometryPreset, str, float | None, float | None, float | None]]]:
    """Generate synthetic trials then delegate to :func:`_run_sweep_from_trial_jobs`."""

    trial_jobs = _prepare_trials(n, trials, seed0, geometries)
    return _run_sweep_from_trial_jobs(
        d=d,
        dispatch=dispatch,
        metric=metric,
        gamma_values=gamma_values,
        tau_values=tau_values,
        geometries=geometries,
        shadows=shadows,
        ad_duals=ad_duals,
        ad_mapper=ad_mapper,
        trial_jobs=trial_jobs,
        progress=progress,
    )


def _plot_heatmaps(
    heatmap: dict[str, dict[str, np.ndarray]],
    gamma_values: Sequence[float],
    tau_values: Sequence[float],
    metric: str,
    geometries: Sequence[GeometryPreset],
    shadows: Sequence[str],
) -> plt.Figure:
    """Render the gamma/tau heatmaps for every geometry/shadow combination."""

    rows = len(shadows)
    cols = len(geometries)
    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        figsize=(4 * cols, 3 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    if rows > 1 and cols == 1:
        axes = axes.T

    gamma_labels = [_format_value(val) for val in gamma_values]
    tau_labels = [_format_value(val) for val in tau_values]

    mesh = None
    for col, geom in enumerate(geometries):
        for row, shadow in enumerate(shadows):
            ax = axes[row, col]
            data = heatmap[geom.name][shadow]
            mesh = ax.imshow(data, origin="lower", aspect="auto")

            if row == 0:
                ax.set_title(geom.label)
            if col == 0:
                ax.set_ylabel(shadow.upper())

            ax.set_xticks(range(len(gamma_values)))
            if row == rows - 1:
                ax.set_xticklabels(gamma_labels)
            else:
                ax.set_xticklabels([])

            ax.set_yticks(range(len(tau_values)))
            if col == 0:
                ax.set_yticklabels(tau_labels)
            else:
                ax.set_yticklabels([])

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.85)
        cbar.set_label(f"mean {metric}")

    fig.supxlabel("gamma")
    fig.supylabel("tau")
    return fig


def _write_csv(records: Sequence[dict], path: str | None) -> None:
    """Persist aggregated results to CSV when a path is supplied."""

    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "geometry",
        "shadow",
        "gamma",
        "tau",
        "dispatch",
        "metric",
        "mean",
        "std",
        "trials",
        "valid_trials",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for the sweep utility."""

    parser = argparse.ArgumentParser(
        description=(
            "Sweep gamma/tau adjustments for SHADOW potentials across geometry presets "
            "and summarize the chosen metric across trials."
        )
    )
    parser.add_argument("--n", type=int, default=100, help="Number of jobs per trial")
    parser.add_argument("--d", type=float, default=2.0, help="Time window parameter d")
    parser.add_argument("--trials", type=int, default=5, help="Trials per gamma/tau pair")
    parser.add_argument(
        "--seed0",
        type=int,
        default=0,
        help="Starting RNG seed (seed, seed+1, ... are used per trial)",
    )
    parser.add_argument(
        "--dispatch",
        choices=["greedy", "greedy+", "batch", "batch+", "rbatch", "rbatch+"],
        default="greedy",
        help=(
            "Dispatch policy to evaluate. '+ variants apply late-arrival shadow "
            "weighting that subtracts only the later job's shadow."
        ),
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
        "--geometries",
        default=",".join(DEFAULT_GEOMETRIES),
        help=(
            "Comma list of geometry presets to evaluate. Choices: "
            + ", ".join(GEOMETRY_PRESETS)
        ),
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
        "--show",
        action="store_true",
        help="Display the generated heatmaps interactively",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the gamma/tau sweep CLI."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.n <= 1:
        parser.error("--n must be greater than one to generate pooling jobs")
    if args.trials <= 0:
        parser.error("--trials must be a positive integer")

    try:
        gamma_values = _parse_values(args.gamma_values)
    except ValueError as exc:  # pragma: no cover - defensive argument parsing
        parser.error(f"invalid --gamma-values: {exc}")
    try:
        tau_values = _parse_values(args.tau_values)
    except ValueError as exc:  # pragma: no cover - defensive argument parsing
        parser.error(f"invalid --tau-values: {exc}")

    try:
        geometry_names = _parse_name_list(args.geometries)
    except ValueError as exc:
        parser.error(f"invalid --geometries: {exc}")
    geometries: list[GeometryPreset] = []
    seen_geometries: set[str] = set()
    for name in geometry_names:
        preset = GEOMETRY_PRESETS.get(name)
        if preset is None:
            parser.error(f"unknown geometry '{name}' (choices: {', '.join(GEOMETRY_PRESETS)})")
        if name in seen_geometries:
            continue
        geometries.append(preset)
        seen_geometries.add(name)
    if not geometries:
        parser.error("at least one geometry must be selected")

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

    ad_duals: AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray | None = None
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
        parser.error("--ad-mapping can only be used when evaluating AD shadows")

    try:
        heatmap, records, best_entries = _run_sweep(
            n=args.n,
            d=float(args.d),
            trials=args.trials,
            seed0=args.seed0,
            dispatch=args.dispatch,
            metric=args.metric,
            gamma_values=gamma_values,
            tau_values=tau_values,
            geometries=geometries,
            shadows=shadows,
            ad_duals=ad_duals,
            ad_mapper=ad_mapper,
        )
    except AverageDualError as exc:
        parser.error(str(exc))

    print("\nBest configurations per geometry/shadow:")
    for geom, shadow, best_mean, best_gamma, best_tau in best_entries:
        label = f"[{geom.label}] {shadow.upper()}"
        if best_gamma is None or best_tau is None or best_mean is None:
            print(f"  {label}: no valid {args.metric} values")
        else:
            print(
                f"  {label}: mean {args.metric}={best_mean:.3f} at gamma={_format_value(best_gamma)}, "
                f"tau={_format_value(best_tau)}",
            )

    fig = _plot_heatmaps(
        heatmap,
        gamma_values,
        tau_values,
        args.metric,
        geometries,
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
