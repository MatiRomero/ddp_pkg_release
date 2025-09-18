"""Gamma/Tau sweeps for SHADOW×DISPATCH configurations across geometries."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ddp.model import Job, generate_jobs
from ddp.scripts.run import run_instance


SHADOWS: tuple[str, ...] = ("naive", "pb", "hd")


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


GEOMETRIES: tuple[GeometryPreset, ...] = (
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


def _prepare_trials(n: int, trials: int, seed0: int) -> list[tuple[int, dict[str, list[Job]]]]:
    """Generate base jobs and geometry variants for each trial seed."""

    prepared: list[tuple[int, dict[str, list[Job]]]] = []
    for offset in range(trials):
        seed = seed0 + offset
        rng = np.random.default_rng(seed)
        base_jobs = generate_jobs(n, rng)
        variants = {preset.name: preset.transform(base_jobs) for preset in GEOMETRIES}
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
    progress: bool = True,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict], list[tuple[GeometryPreset, str, float | None, float | None, float | None]]]:
    """Execute the gamma/tau sweep and return heatmap data, records, and best combos."""

    trial_jobs = _prepare_trials(n, trials, seed0)

    heatmap: dict[str, dict[str, np.ndarray]] = {
        geom.name: {
            shadow: np.full((len(tau_values), len(gamma_values)), np.nan, dtype=float)
            for shadow in SHADOWS
        }
        for geom in GEOMETRIES
    }

    records: list[dict] = []
    best_entries: list[tuple[GeometryPreset, str, float | None, float | None, float | None]] = []

    total_trials = (
        len(GEOMETRIES)
        * len(SHADOWS)
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
        for geom in GEOMETRIES:
            for shadow in SHADOWS:
                best_mean: float | None = None
                best_gamma: float | None = None
                best_tau: float | None = None

                for tau_index, tau in enumerate(tau_values):
                    for gamma_index, gamma in enumerate(gamma_values):
                        if show_progress:
                            progress_bar.set_description(
                                f"{geom.name}/{shadow} γ={_format_value(float(gamma))} τ={_format_value(float(tau))}"
                            )
                        values: list[float] = []

                        for seed, job_variants in trial_jobs:
                            jobs = job_variants[geom.name]
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
                            )
                            row = result["rows"][0]
                            metric_value = _extract_metric(row, metric)
                            if metric_value is not None:
                                values.append(metric_value)
                            progress_bar.update()

                        if values:
                            mean_val = statistics.fmean(values)
                            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
                            heatmap[geom.name][shadow][tau_index, gamma_index] = mean_val
                            if best_mean is None or mean_val > best_mean:
                                best_mean = mean_val
                                best_gamma = float(gamma)
                                best_tau = float(tau)
                        else:
                            mean_val = float("nan")
                            std_val = float("nan")

                        records.append(
                            {
                                "geometry": geom.name,
                                "shadow": shadow,
                                "gamma": float(gamma),
                                "tau": float(tau),
                                "dispatch": dispatch,
                                "metric": metric,
                                "mean": mean_val,
                                "std": std_val,
                                "trials": trials,
                                "valid_trials": len(values),
                            }
                        )

                best_entries.append((geom, shadow, best_mean, best_gamma, best_tau))
    finally:
        progress_bar.close()

    return heatmap, records, best_entries


def _plot_heatmaps(
    heatmap: dict[str, dict[str, np.ndarray]],
    gamma_values: Sequence[float],
    tau_values: Sequence[float],
    metric: str,
) -> plt.Figure:
    """Render the gamma/tau heatmaps for every geometry/shadow combination."""

    rows = len(SHADOWS)
    cols = len(GEOMETRIES)
    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        figsize=(4 * cols, 3 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    gamma_labels = [_format_value(val) for val in gamma_values]
    tau_labels = [_format_value(val) for val in tau_values]

    mesh = None
    for col, geom in enumerate(GEOMETRIES):
        for row, shadow in enumerate(SHADOWS):
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

    heatmap, records, best_entries = _run_sweep(
        n=args.n,
        d=float(args.d),
        trials=args.trials,
        seed0=args.seed0,
        dispatch=args.dispatch,
        metric=args.metric,
        gamma_values=gamma_values,
        tau_values=tau_values,
    )

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

    fig = _plot_heatmaps(heatmap, gamma_values, tau_values, args.metric)

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
