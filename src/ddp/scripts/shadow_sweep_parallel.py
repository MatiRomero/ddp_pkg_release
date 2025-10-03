"""Parallel gamma/tau sweep driver for SHADOW×DISPATCH experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from ddp.model import Job
from ddp.scripts.run import (
    AverageDualError,
    AverageDualTable,
    load_average_dual_mapper,
    load_average_duals,
)
from ddp.scripts.shadow_sweep import (
    ALL_SHADOWS,
    DEFAULT_GEOMETRIES,
    DEFAULT_SHADOWS,
    GEOMETRY_PRESETS,
    GeometryPreset,
    SweepCombinationResult,
    build_arg_parser,
    evaluate_sweep_combination,
    _format_value,
    _parse_name_list,
    _parse_values,
    _plot_heatmaps,
    _prepare_trials,
    _write_csv,
)


def _evaluate_combination_worker(
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
    | None,
) -> SweepCombinationResult:
    """Worker entrypoint that delegates to :func:`evaluate_sweep_combination`."""

    # ``evaluate_sweep_combination`` handles aggregation; workers simply forward args.
    return evaluate_sweep_combination(
        geometry=geometry,
        shadow=shadow,
        gamma=gamma,
        tau=tau,
        d=d,
        dispatch=dispatch,
        metric=metric,
        trial_jobs=trial_jobs,
        ad_duals=ad_duals,
        ad_mapper=ad_mapper,
        trial_ad_duals=trial_ad_duals,
        progress=None,
    )


def _build_geometry_trials(
    trial_jobs: Sequence[tuple[int, Mapping[str, Sequence[Job]]]],
    geometries: Sequence[GeometryPreset],
) -> dict[str, list[tuple[int, Sequence[Job]]]]:
    """Extract geometry-specific job lists for every trial."""

    return {
        geom.name: [(seed, job_variants[geom.name]) for seed, job_variants in trial_jobs]
        for geom in geometries
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the parallel gamma/tau sweep CLI."""

    parser = build_arg_parser()
    parser.description += " (parallel execution variant)"
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes (defaults to CPU count). Set to 1 for serial "
            "execution."
        ),
    )
    args = parser.parse_args(argv)

    if args.n <= 1:
        parser.error("--n must be greater than one to generate pooling jobs")
    if args.trials <= 0:
        parser.error("--trials must be a positive integer")
    if args.workers < 0:
        parser.error("--workers must be non-negative")

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
            parser.error(
                f"unknown geometry '{name}' (choices: {', '.join(GEOMETRY_PRESETS)})"
            )
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
                ad_mapper_callable = load_average_dual_mapper(args.ad_mapping)
            except (ModuleNotFoundError, AttributeError, ValueError, TypeError) as exc:
                parser.error(f"failed to resolve --ad-mapping: {exc}")
            ad_mapper = ad_mapper_callable
        if (
            isinstance(ad_duals, AverageDualTable)
            and ad_duals.by_job is None
            and ad_duals.by_type is not None
            and ad_mapper is None
        ):
            parser.error("type-indexed average-dual tables require --ad-mapping")
    elif args.ad_mapping:
        parser.error("--ad-mapping can only be used when evaluating AD shadows")

    max_workers: int | None = args.workers or None
    d_value = float(args.d)

    trial_jobs = _prepare_trials(args.n, args.trials, args.seed0, geometries)
    geometry_trial_jobs = _build_geometry_trials(trial_jobs, geometries)

    heatmap: dict[str, dict[str, np.ndarray]] = {
        geom.name: {
            shadow: np.full((len(tau_values), len(gamma_values)), np.nan, dtype=float)
            for shadow in shadows
        }
        for geom in geometries
    }

    total_trials = (
        len(geometries)
        * len(shadows)
        * len(tau_values)
        * len(gamma_values)
        * len(trial_jobs)
    )
    show_progress = total_trials > 0
    progress_bar = tqdm(
        total=total_trials,
        desc="Trials",
        unit="trial",
        disable=not show_progress,
    )

    try:
        results: dict[tuple[str, str, int, int], SweepCombinationResult] = {}
        future_to_meta: dict = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for geom in geometries:
                geom_trials = geometry_trial_jobs[geom.name]
                for shadow in shadows:
                    for tau_index, tau in enumerate(tau_values):
                        for gamma_index, gamma in enumerate(gamma_values):
                            future = executor.submit(
                                _evaluate_combination_worker,
                                geom.name,
                                shadow,
                                float(gamma),
                                float(tau),
                                d_value,
                                args.dispatch,
                                args.metric,
                                geom_trials,
                                ad_duals,
                                ad_mapper,
                                None,
                            )
                            future_to_meta[future] = (
                                geom,
                                shadow,
                                tau_index,
                                gamma_index,
                                float(gamma),
                                float(tau),
                            )

            for future in as_completed(future_to_meta):
                geom, shadow, tau_index, gamma_index, gamma_val, tau_val = future_to_meta[future]
                try:
                    combo_result = future.result()
                except AverageDualError as exc:
                    for pending in future_to_meta:
                        if pending is not future:
                            pending.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise exc
                if show_progress:
                    progress_bar.set_description(
                        f"{geom.name}/{shadow} γ={_format_value(gamma_val)} τ={_format_value(tau_val)}"
                    )
                progress_bar.update(combo_result.trials)
                heatmap[geom.name][shadow][tau_index, gamma_index] = combo_result.mean
                results[(geom.name, shadow, tau_index, gamma_index)] = combo_result
    except AverageDualError as exc:
        parser.error(str(exc))
    finally:
        progress_bar.close()

    records: list[dict] = []
    best_entries: list[tuple[GeometryPreset, str, float | None, float | None, float | None]] = []
    trials = len(trial_jobs)

    for geom in geometries:
        for shadow in shadows:
            best_mean: float | None = None
            best_gamma: float | None = None
            best_tau: float | None = None
            for tau_index, tau in enumerate(tau_values):
                for gamma_index, gamma in enumerate(gamma_values):
                    combo_result = results[(geom.name, shadow, tau_index, gamma_index)]
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
                            "dispatch": args.dispatch,
                            "metric": args.metric,
                            "mean": combo_result.mean,
                            "std": combo_result.std,
                            "trials": trials,
                            "valid_trials": combo_result.valid_trials,
                        }
                    )
            best_entries.append((geom, shadow, best_mean, best_gamma, best_tau))

    print("\nBest configurations per geometry/shadow:")
    for geom, shadow, best_mean, best_gamma, best_tau in best_entries:
        label = f"[{geom.label}] {shadow.upper()}"
        if best_gamma is None or best_tau is None or best_mean is None:
            print(f"  {label}: no valid {args.metric} values")
        else:
            print(
                f"  {label}: mean {args.metric}={best_mean:.3f} at gamma={_format_value(best_gamma)}, "
                f"tau={_format_value(best_tau)}"
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
