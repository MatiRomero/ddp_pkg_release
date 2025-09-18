"""Minimal working example focused on debugging the ``rbatch`` dispatcher."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from ddp.algorithms.potential import potential as potential_vec
from ddp.engine.opt import max_weight_matching_subset
from ddp.model import Job
from ddp.scripts.run import make_weight_fn, reward_fn, run_instance


def _format_job(job: Job, due_time: float, idx: int) -> str:
    origin = f"({job.origin[0]:>4.1f}, {job.origin[1]:>4.1f})"
    dest = f"({job.dest[0]:>4.1f}, {job.dest[1]:>4.1f})"
    return (
        f"{idx:>2d}  ts={job.timestamp:>4.1f}  due={due_time:>4.1f}  "
        f"{origin} -> {dest}  length={job.length:>5.2f}"
    )


def _shadow_vector(label: str, jobs: Iterable[Job]) -> np.ndarray:
    jobs_list = list(jobs)
    n = len(jobs_list)
    if label == "naive":
        return np.zeros(n, dtype=float)
    if label == "pb":
        lengths = np.array([job.length for job in jobs_list], dtype=float)
        return potential_vec(lengths)
    msg = f"Unsupported shadow '{label}' in mwe_02"
    raise ValueError(msg)


def debug_rbatch(jobs: list[Job], d: float, shadow: str) -> None:
    """Print the event-by-event behaviour of ``rbatch`` for ``shadow``."""

    timestamps = np.array([job.timestamp for job in jobs], dtype=float)
    due_time = timestamps + float(d)
    event_times = np.unique(np.concatenate([timestamps, due_time]))

    sp = _shadow_vector(shadow, jobs)
    weight_fn = make_weight_fn(reward_fn, sp)

    available: set[int] = set()
    arrived = np.zeros(len(jobs), dtype=bool)

    print(f"\n[rbatch debug] shadow={shadow}")
    for t in event_times:
        to_add = np.where((~arrived) & (timestamps <= t))[0]
        for idx in to_add:
            available.add(int(idx))
            arrived[idx] = True

        due_now = sorted(i for i in available if due_time[i] <= t)
        if not due_now:
            continue

        print(
            f"  t={t:>4.1f}  available={sorted(available)}  due_now={due_now}"
        )

        for i in due_now:
            if i not in available:
                continue

            critical = {i}

            def w_eff(a: int, b: int, _: list[Job]) -> float:
                w = weight_fn(a, b, jobs)
                if a in critical:
                    w += float(sp[a])
                if b in critical:
                    w += float(sp[b])
                return w

            matching = max_weight_matching_subset(
                list(available),
                jobs,
                reward_fn,
                weight_fn=w_eff,
                method="auto",
            )
            pairs = matching["pairs"]
            readable = [
                (int(u), int(v), float(w)) for (u, v, w) in pairs
            ]
            print(f"    matching={readable}")

            partner = None
            weight = None
            for u, v, w_val in pairs:
                a, b = (u, v) if int(u) < int(v) else (v, u)
                if int(a) == i:
                    partner = int(b)
                    weight = float(w_val)
                    break
                if int(b) == i:
                    partner = int(a)
                    weight = float(w_val)
                    break

            if partner is not None and partner in available:
                reward = reward_fn(i, partner, jobs)
                print(
                    f"    dispatch pair=({i}, {partner})  weight={weight:6.2f}"
                    f"  reward={reward:6.2f}"
                )
                available.discard(i)
                available.discard(partner)
            else:
                print(f"    solo job={i}")
                available.discard(i)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal working example tailored to observing the rbatch loop."
        )
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib plots for the resulting matches.",
    )
    args = parser.parse_args(argv)

    plot = bool(args.plot)

    # Hand-crafted instance: jobs clustered into spatial pairs with staggered
    # arrival times so that the rbatch dispatcher needs to resolve the matching
    # multiple times as deadlines are hit.
    job_specs = [
        # (origin, destination, timestamp)
        ((0.0, 0.0), (10.0, 0.0), 0.0),
        ((1.2, 0.0), (11.2, 0.0), 0.3),
        ((20.0, 0.0), (30.0, 0.0), 0.0),
        ((21.2, 0.0), (31.2, 0.0), 0.4),
        ((0.0, 10.0), (8.0, 10.0), 0.6),
        ((0.8, 10.0), (8.8, 10.0), 0.9),
        ((20.0, 10.0), (28.0, 10.0), 1.1),
        ((20.8, 10.0), (28.8, 10.0), 1.4),
    ]

    jobs = [
        Job(origin=origin, dest=dest, timestamp=timestamp)
        for origin, dest, timestamp in job_specs
    ]
    d = 3.0

    timestamps = np.array([job.timestamp for job in jobs], dtype=float)
    due_time = timestamps + d
    print("Instance summary (n=8, d=3):")
    for idx, job in enumerate(jobs):
        print(_format_job(job, due_time[idx], idx))

    shadows = ("naive", "pb")
    dispatches = ("greedy", "batch", "rbatch")

    result = run_instance(
        jobs=jobs,
        d=d,
        shadows=shadows,
        dispatches=dispatches,
        seed=0,
        with_opt=True,
        save_csv="",
        print_table=True,
        return_details=True,
        print_matches=False,
    )

    details = result.get("details", {})
    print("\nFinal matches per algorithm (pairs i-j) with arrivals and solos:")
    for sh in shadows:
        for disp in dispatches:
            info = details.get((sh, disp))
            if not info:
                continue
            pairs = info["pairs"]
            solos = info["solos"]
            print(f"{sh.upper():<10} {disp:<12} pairs={pairs}  solos={solos}")

    for sh in shadows:
        debug_rbatch(jobs, d, sh)

    if not plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        msg = (
            "Matplotlib is required for plotting. Install the optional dependencies "
            "with `pip install ddp[plot]` or, for editable installs, `pip install -e .[plot]`."
        )
        raise SystemExit(msg) from exc

    theta_arr = np.array([job.length for job in result["jobs"]], dtype=float)
    t_arr = result["timestamps"]

    palette = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    )

    def alg_color(sh: str, disp: str) -> str:
        i = list(shadows).index(sh)
        j = list(dispatches).index(disp)
        return palette[(i * len(dispatches) + j) % len(palette)]

    def plot_one(sh: str, disp: str) -> None:
        info = details.get((sh, disp))
        if not info:
            return
        pairs = info["pairs"]
        col = alg_color(sh, disp)

        plt.figure()
        plt.scatter(t_arr, theta_arr, s=50, label="jobs", color="#666666", alpha=0.8)
        for (i, j) in pairs:
            plt.plot(
                [t_arr[i], t_arr[j]],
                [theta_arr[i], theta_arr[j]],
                linewidth=2,
                alpha=0.95,
                color=col,
            )
        plt.xlabel("arrival / period")
        plt.ylabel("job length")
        plt.title(f"{sh.upper()} + {disp}")
        plt.tight_layout()

    for sh in shadows:
        for disp in dispatches:
            plot_one(sh, disp)

    plt.show()


if __name__ == "__main__":
    main()
