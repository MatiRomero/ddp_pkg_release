"""Trace the available set as a dispatch policy processes a single instance."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from ddp.algorithms.potential import potential as potential_vec
from ddp.engine.opt import compute_lp_relaxation
from ddp.engine.sim import simulate
from ddp.model import Job
from ddp.scripts.run import (
    make_local_score,
    make_weight_fn,
    make_weight_fn_latest_shadow,
    reward_fn,
)


def _load_jobs(path: str) -> list[Job]:
    with np.load(path) as data:
        try:
            origins = data["origins"]
            dests = data["dests"]
            timestamps = data["timestamps"]
        except KeyError as exc:
            msg = "--jobs file must contain 'origins', 'dests', and 'timestamps'"
            raise SystemExit(msg) from exc

    if not (len(origins) == len(dests) == len(timestamps)):
        raise SystemExit("Mismatched job array lengths in --jobs")

    jobs = [
        Job(
            origin=tuple(map(float, origin)),
            dest=tuple(map(float, dest)),
            timestamp=float(ts),
        )
        for origin, dest, ts in zip(origins, dests, timestamps)
    ]
    return jobs


def _shadow_vector(label: str, jobs: Sequence[Job], time_window) -> np.ndarray:
    jobs_list = list(jobs)
    n = len(jobs_list)
    if label == "naive":
        return np.zeros(n, dtype=float)
    if label == "pb":
        lengths = np.array([job.length for job in jobs_list], dtype=float)
        return potential_vec(lengths)
    if label == "hd":
        lp = compute_lp_relaxation(jobs_list, reward_fn, time_window=time_window)
        return np.array(lp["duals"], dtype=float)
    raise SystemExit(f"Unsupported shadow '{label}'.")


def _prepare_dispatch(
    policy: str,
    base_shadow: np.ndarray,
    gamma: float,
    tau: float,
    gamma_plus: float | None,
    tau_plus: float | None,
) -> tuple[str, str, callable, callable | None, np.ndarray | None]:
    """Return dispatch configuration for :func:`simulate`.

    Output is ``(decision_rule, sim_policy, score_fn, weight_fn, sim_shadow)``. ``sim_shadow``
    is passed to :func:`simulate` for critical adjustments (BATCH/RBATCH).
    """

    sp = np.array(base_shadow, dtype=float, copy=True)
    if policy in {"batch+", "rbatch+"}:
        gamma_eff = gamma_plus if gamma_plus is not None else 1.0
        tau_eff = tau_plus if tau_plus is not None else 0.0
        sp = sp * gamma_eff + tau_eff
        weight_fn = make_weight_fn_latest_shadow(reward_fn, sp)
        sim_shadow: np.ndarray | None = None
    else:
        sp = sp * gamma + tau
        weight_fn = make_weight_fn(reward_fn, sp)
        sim_shadow = sp

    score_fn = make_local_score(reward_fn, sp)

    if policy == "greedy":
        return "naive", "score", score_fn, None, None
    if policy == "greedy+":
        return "threshold", "score", score_fn, None, None
    if policy == "batch":
        return "policy", "batch", score_fn, weight_fn, sim_shadow
    if policy == "batch+":
        return "policy", "batch", score_fn, weight_fn, sim_shadow
    if policy == "rbatch":
        return "policy", "rbatch", score_fn, weight_fn, sim_shadow
    if policy == "rbatch+":
        return "policy", "rbatch", score_fn, weight_fn, sim_shadow
    raise SystemExit(f"Unsupported policy '{policy}'.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a single instance under one dispatch policy and trace the available set."
        )
    )
    parser.add_argument("--jobs", required=True, help="Path to .npz instance")
    parser.add_argument("--d", type=float, required=True, help="Dispatch window (scalar)")
    parser.add_argument(
        "--policy",
        required=True,
        choices=["greedy", "greedy+", "batch", "batch+", "rbatch", "rbatch+"],
        help=(
            "Dispatch policy to trace. The '+ variants use late-arrival shadow "
            "weighting with reward(i, j) - s_late (subtracting only the later "
            "job's shadow value)."
        ),
    )
    parser.add_argument(
        "--shadow",
        default="naive",
        choices=["naive", "pb", "hd"],
        help="Shadow potential family used to build scores/weights.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help=(
            "Scale factor applied to the shadow potentials before dispatch. "
            "Defaults to 0.5 for compatibility with the batch/rbatch heuristics."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="Additive offset applied to the base shadow potentials before dispatch.",
    )
    parser.add_argument(
        "--plus_gamma",
        type=float,
        default=1.0,
        help=(
            "Scale factor for the late-arrival ('+' variants) shadow potentials. "
            "Defaults to 1 so the weight is reward(i, j) - s_late unless overridden."
        ),
    )
    parser.add_argument(
        "--plus_tau",
        type=float,
        default=None,
        help=(
            "Additive offset for the late-arrival ('+' variants) shadow potentials. "
            "Defaults to 0 when omitted."
        ),
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the size of the available set over the event timeline.",
    )
    parser.add_argument(
        "--tie_breaker",
        default="distance",
        choices=["distance", "random"],
        help=(
            "Tie-breaking rule used by greedy dispatch when multiple partners share the "
            "same score."
        ),
    )
    args = parser.parse_args(argv)

    jobs = _load_jobs(args.jobs)
    if not jobs:
        raise SystemExit("Instance contains no jobs.")

    time_window = float(args.d)

    base_shadow = _shadow_vector(args.shadow, jobs, time_window)

    decision_rule, sim_policy, score_fn, weight_fn, sim_shadow = _prepare_dispatch(
        args.policy,
        base_shadow,
        args.gamma,
        args.tau,
        args.plus_gamma,
        args.plus_tau,
    )

    event_log: list[tuple[float, str, int]] = []

    def hook(time: float, available: Sequence[int], due_now: Sequence[int], phase: str) -> None:
        avail_list = list(available)
        due_list = list(due_now)
        print(
            f"t={time:>6.2f}  phase={phase:<6}  available={avail_list} "
            f"(size={len(avail_list)})  due_now={due_list}"
        )
        event_log.append((float(time), phase, len(avail_list)))

    print(
        f"Tracing policy={args.policy} with shadow={args.shadow} on {len(jobs)} jobs (d={time_window})."
    )

    result = simulate(
        jobs,
        score_fn,
        reward_fn,
        decision_rule,
        time_window=time_window,
        policy=sim_policy,
        weight_fn=weight_fn,
        shadow=sim_shadow,
        seed=0,
        tie_breaker=args.tie_breaker,
        event_hook=hook,
    )

    print("\nDispatch outcome:")
    pairs = [(int(i), int(j)) for (i, j, *_rest) in result["pairs"]]
    print(f"  pairs={pairs}")
    print(f"  solos={list(result['solos'])}")
    print(f"  total_savings={result['total_savings']:.3f}")
    print(f"  pooled_pct={result['pooled_pct']:.1f}%")

    if not args.plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        msg = (
            "Matplotlib is required for plotting. Install optional dependencies with "
            "`pip install ddp[plot]` or `pip install -e .[plot]`."
        )
        raise SystemExit(msg) from exc

    if not event_log:
        print("No events recorded; skipping plot.")
        return

    times = [t for (t, _phase, _count) in event_log]
    counts = [c for (_t, _phase, c) in event_log]

    plt.figure()
    plt.step(times, counts, where="post", label="|available|")
    plt.scatter(times, counts, color="C1", s=30, zorder=3)
    plt.xlabel("event time")
    plt.ylabel("available jobs")
    plt.title(
        f"Available set size — policy={args.policy}, shadow={args.shadow}, d={time_window}"
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

