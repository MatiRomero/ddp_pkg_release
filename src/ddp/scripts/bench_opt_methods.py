#!/usr/bin/env python3
"""Benchmark helpers for the ``--with_opt`` backends.

This script generates random job instances under several geometric layouts and
times the available backends used by :func:`ddp.engine.opt.compute_opt`.  It is
intended to provide a quick comparison between the ``auto``, ``networkx`` and
``ilp`` methods when ``--with_opt`` is enabled in the experiment runners.

Usage (as a module)::

    python -m ddp.scripts.bench_opt_methods --n 250 --trials 5

The output prints a compact table with mean runtime (in milliseconds) for each
method across the requested trials, along with the backend that ``auto`` chose
internally.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ddp.engine.opt import compute_opt
from ddp.model import Job, generate_jobs, reward as pooling_reward


MethodName = str


def _reward_fn(i: int, j: int, jobs: Sequence[Job]) -> float:
    """Wrapper matching :func:`ddp.scripts.run.reward_fn`."""

    return pooling_reward([jobs[i], jobs[j]])


@dataclass(frozen=True)
class Scenario:
    """Configuration describing how jobs should be transformed for a trial."""

    name: str
    description: str
    fix_origin_zero: bool
    flatten_axis: str | None


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="1d_common_origin",
        description=(
            "All origins fixed at the depot with coordinates flattened onto a single axis."
        ),
        fix_origin_zero=True,
        flatten_axis="x",
    ),
    Scenario(
        name="2d_common_origin",
        description="All origins fixed at the depot while destinations remain 2-D.",
        fix_origin_zero=True,
        flatten_axis=None,
    ),
    Scenario(
        name="2d_different_origins",
        description="Fully random origins and destinations in 2-D.",
        fix_origin_zero=False,
        flatten_axis=None,
    ),
)


def _flatten(point: Iterable[float], axis: int) -> tuple[float, float]:
    coords = [float(point[0]), float(point[1])]
    coords[axis] = 0.0
    return coords[0], coords[1]


def _apply_geometry(jobs: Sequence[Job], scenario: Scenario) -> list[Job]:
    """Apply geometry tweaks (origin pinning / axis flattening) to ``jobs``."""

    transformed = list(jobs)
    if scenario.fix_origin_zero:
        transformed = [
            Job(origin=(0.0, 0.0), dest=job.dest, timestamp=job.timestamp)
            for job in transformed
        ]

    if scenario.flatten_axis is not None:
        axis = 0 if scenario.flatten_axis.lower() == "x" else 1
        transformed = [
            Job(
                origin=_flatten(job.origin, axis),
                dest=_flatten(job.dest, axis),
                timestamp=job.timestamp,
            )
            for job in transformed
        ]

    return transformed


def _generate_jobs(n: int, rng: np.random.Generator, scenario: Scenario) -> list[Job]:
    jobs = generate_jobs(n, rng)
    return _apply_geometry(jobs, scenario)


def _benchmark_scenario(
    *,
    n: int,
    trials: int,
    seed0: int,
    scenario: Scenario,
    methods: Sequence[MethodName],
) -> tuple[dict[MethodName, list[float]], dict[MethodName, str | None], dict[MethodName, Counter], list[dict]]:
    timings: dict[MethodName, list[float]] = {method: [] for method in methods}
    errors: dict[MethodName, str | None] = {method: None for method in methods}
    backend_counts: dict[MethodName, Counter] = {method: Counter() for method in methods}
    mismatches: list[dict] = []

    for trial in range(trials):
        print(
            f"Scenario {scenario.name}: trial {trial + 1}/{trials}",
            flush=True,
        )
        rng = np.random.default_rng(seed0 + trial)
        jobs = _generate_jobs(n, rng, scenario)

        baseline_total = None
        baseline_method = None

        for method in methods:
            if errors[method] is not None:
                continue  # Skip once a persistent failure occurs.

            start = time.perf_counter()
            try:
                result = compute_opt(jobs, _reward_fn, method=method)
            except Exception as exc:  # pragma: no cover - depends on optional deps
                errors[method] = str(exc)
                continue

            elapsed = time.perf_counter() - start
            timings[method].append(elapsed)

            backend = result.get("method", method)
            backend_counts[method][backend] += 1

            total = float(result.get("total_reward", 0.0))
            if baseline_total is None:
                baseline_total = total
                baseline_method = method
            elif abs(total - baseline_total) > 1e-6:
                mismatches.append(
                    {
                        "trial": trial,
                        "method": method,
                        "total": total,
                        "baseline_total": baseline_total,
                        "baseline_method": baseline_method,
                    }
                )

    return timings, errors, backend_counts, mismatches


def _format_backend_counts(counter: Counter) -> str:
    if not counter:
        return ""
    parts = [f"{name}×{count}" for name, count in counter.most_common()]
    return ", ".join(parts)


def _summarize_results(
    *,
    scenario: Scenario,
    timings: dict[MethodName, list[float]],
    errors: dict[MethodName, str | None],
    backend_counts: dict[MethodName, Counter],
    mismatches: Sequence[dict],
    methods: Sequence[MethodName],
) -> None:
    print(f"\n=== {scenario.name} ===")
    print(scenario.description)

    for method in methods:
        times = timings[method]
        err = errors[method]
        label = method.lower()

        if not times:
            status = f"failed ({err})" if err else "no successful runs"
            print(f"  {label:<12} -> {status}")
            continue

        mean_s = statistics.mean(times)
        std_s = statistics.stdev(times) if len(times) > 1 else 0.0
        backend_note = _format_backend_counts(backend_counts[method])

        print(
            f"  {label:<12} -> mean={mean_s * 1e3:8.3f} ms"
            f" | std={std_s * 1e3:8.3f} ms"
            f" | min={min(times) * 1e3:8.3f} ms"
            f" | max={max(times) * 1e3:8.3f} ms"
            f" | runs={len(times)}"
        )
        if backend_note:
            print(f"      backends: {backend_note}")
        if err is not None:
            print(f"      note: encountered error after {len(times)} runs -> {err}")

    if mismatches:
        print("  WARNING: mismatch in total_reward between methods detected")
        for item in mismatches:
            print(
                "    trial={trial} baseline={baseline_method} total={baseline_total:.6f}"
                " vs {method} total={total:.6f}".format(**item)
            )


def _parse_methods(text: str) -> list[MethodName]:
    allowed = {"auto", "networkx", "ilp"}
    methods = [token.strip().lower() for token in text.split(",") if token.strip()]
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        msg = f"Unsupported method(s): {', '.join(sorted(set(unknown)))}"
        raise ValueError(msg)
    if not methods:
        raise ValueError("At least one method must be provided")
    return methods


def _parse_scenarios(text: str) -> tuple[Scenario, ...]:
    if text.strip().lower() in {"", "all"}:
        return SCENARIOS

    lookup = {scenario.name: scenario for scenario in SCENARIOS}
    names = [token.strip() for token in text.split(",") if token.strip()]
    missing = [name for name in names if name not in lookup]
    if missing:
        msg = f"Unknown scenario name(s): {', '.join(sorted(set(missing)))}"
        raise ValueError(msg)
    return tuple(lookup[name] for name in names)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the compute_opt backends used for --with_opt under several"
            " geometric layouts."
        )
    )
    parser.add_argument("--n", type=int, default=200, help="Number of jobs per trial")
    parser.add_argument("--trials", type=int, default=3, help="Trials per scenario")
    parser.add_argument(
        "--seed0",
        type=int,
        default=0,
        help="Base RNG seed; each trial uses seed0 + trial index",
    )
    parser.add_argument(
        "--methods",
        default="auto,networkx,ilp",
        help="Comma-separated list from {auto, networkx, ilp}",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="Comma list of scenario names (default: all)",
    )

    args = parser.parse_args(argv)

    if args.n <= 1:
        parser.error("--n must be greater than one")
    if args.trials <= 0:
        parser.error("--trials must be positive")

    try:
        methods = _parse_methods(args.methods)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        scenarios = _parse_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        "Benchmarking compute_opt backends for"  # pragma: no cover - CLI output only
        f" n={args.n} over {args.trials} trial(s)"
    )

    for scenario in scenarios:
        timings, errors, backend_counts, mismatches = _benchmark_scenario(
            n=args.n,
            trials=args.trials,
            seed0=args.seed0,
            scenario=scenario,
            methods=methods,
        )
        _summarize_results(
            scenario=scenario,
            timings=timings,
            errors=errors,
            backend_counts=backend_counts,
            mismatches=mismatches,
            methods=methods,
        )


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    main()

