from typing import Sequence

import numpy as np

from ddp.engine.opt import max_weight_matching_subset
from ddp.model import Job, distance


def simulate(
    jobs: Sequence[Job],
    score_fn,           # local score(i,j,theta) = reward - s_j
    reward_fn,          # reward(i,j,theta)
    decision_rule="naive",         # 'naive' or 'threshold' (ignored for batch/rbatch)
    timestamps=None,               # array or None → 1,2,3,...
    time_window=None,              # scalar or array
    policy="score",                # 'score' | 'batch' | 'rbatch' | 'batch2' | 'rbatch2'
    weight_fn=None,                # base weight for matching: reward - s_i - s_j
    shadow=None,                   # vector s_i (needed for critical adjustment)
    seed=0,
    tie_breaker: str = "distance",
    event_hook=None,               # optional callback(time, available, due_now, phase)
    tau_s: float = 30.0,
    dispatch_hook=None,            # optional callback(time, dispatched_indices)
):
    jobs = list(jobs)
    n = len(jobs)

    valid_tie_breakers = {"distance", "random"}
    if tie_breaker not in valid_tie_breakers:
        raise ValueError(
            f"Unsupported tie_breaker '{tie_breaker}'. Choose from {sorted(valid_tie_breakers)}."
        )

    # time inputs
    if timestamps is None:
        timestamps = np.array([job.timestamp for job in jobs], dtype=float)
    else:
        timestamps = np.asarray(timestamps, dtype=float)
        if len(timestamps) != n:
            raise ValueError("len(timestamps) must equal len(jobs)")
    if time_window is None:
        raise ValueError("time_window must be provided (scalar or array-like)")
    if np.isscalar(time_window):
        tw = np.full(n, float(time_window), dtype=float)
    else:
        tw = np.asarray(time_window, dtype=float)
        if len(tw) != n:
            raise ValueError("len(time_window) must equal len(jobs)")
        if np.any(tw < 0):
            raise ValueError("time_window values must be non-negative")

    due_time = timestamps + tw
    periodic_policies = {"batch2", "rbatch2"}
    if policy in periodic_policies:
        if tau_s <= 0:
            raise ValueError("tau_s must be positive for periodic policies")
        if n:
            min_ts = float(np.min(timestamps))
            first_tick = max(float(tau_s), float(np.ceil(min_ts / tau_s) * tau_s))
        else:
            first_tick = float(tau_s)
        max_due = float(np.max(due_time)) if n else float(tau_s)
        tick_stop = max_due + float(tau_s)
        tick_times = np.arange(first_tick, tick_stop + 0.5 * float(tau_s), float(tau_s))
        tick_times = tick_times.astype(float)
    else:
        tick_times = np.array([], dtype=float)

    event_times = np.unique(np.concatenate([timestamps, due_time, tick_times]))
    tick_set = set(float(t) for t in tick_times)
    arrived = np.zeros(n, dtype=bool)
    available = set()
    paired = set()
    total_savings = 0.0
    pairs = []  # (i, j, score_or_weight, reward)
    solos = []
    rng = np.random.default_rng(seed)
    dispatch_times: dict[int, float] = {}

    def _mark_dispatched(indices: Sequence[int], when: float) -> None:
        if not indices:
            return
        ts = float(when)
        for idx in indices:
            int_idx = int(idx)
            if int_idx not in dispatch_times:
                dispatch_times[int_idx] = ts
        if dispatch_hook is not None:
            dispatch_hook(ts, tuple(int(idx) for idx in indices))

    for t in event_times:
        # arrivals
        to_add = np.where((~arrived) & (timestamps <= t))[0]
        for i in to_add:
            available.add(int(i))
            arrived[i] = True

        # due now
        due_now = [i for i in available if due_time[i] <= t]
        if event_hook is not None:
            available_snapshot = tuple(sorted(available))
            due_snapshot = tuple(sorted(due_now))
            event_hook(float(t), available_snapshot, due_snapshot, "before")
        is_tick = policy in periodic_policies and float(t) in tick_set
        if not due_now and not is_tick:
            continue

        if policy in periodic_policies and not is_tick:
            if event_hook is not None:
                available_snapshot = tuple(sorted(available))
                due_snapshot = tuple(sorted(due_now))
                event_hook(float(t), available_snapshot, due_snapshot, "after")
            continue

        # BATCH: solve matching on all available; critical adjustment; dispatch all
        if policy == "batch":
            C = set(due_now)
            def w_eff(i, j, th):
                w = (weight_fn(i, j, th) if weight_fn is not None else reward_fn(i, j, th))
                if shadow is not None:
                    if i in C: w += float(shadow[i])
                    if j in C: w += float(shadow[j])
                return w

            result = max_weight_matching_subset(list(available), jobs, reward_fn, weight_fn=w_eff, method="auto")
            matched = set()
            for (i, j, w_weight) in result["pairs"]:
                r = float(reward_fn(i, j, jobs))
                total_savings += r
                pairs.append((i, j, float(w_weight), r))
                matched.add(i); matched.add(j)
                _mark_dispatched((i, j), t)
            for i in sorted(v for v in available if v not in matched):
                solos.append(i)
                _mark_dispatched((i,), t)
            available.clear()
            if event_hook is not None:
                available_snapshot = tuple(sorted(available))
                due_snapshot = tuple(sorted(i for i in available if due_time[i] <= t))
                event_hook(float(t), available_snapshot, due_snapshot, "after")
            continue

        # RBATCH: per critical i; only dispatch i (and partner)
        if policy == "rbatch":
            for i in sorted(due_now, key=lambda k: (due_time[k], k)):
                if i not in available:
                    continue
                C = {i}
                def w_eff_i(a, b, th):
                    w = (weight_fn(a, b, th) if weight_fn is not None else reward_fn(a, b, th))
                    if shadow is not None:
                        if a in C: w += float(shadow[a])
                        if b in C: w += float(shadow[b])
                    return w

                result = max_weight_matching_subset(list(available), jobs, reward_fn, weight_fn=w_eff_i, method="auto")
                partner = None; w_ij = None
                for (u, v, w) in result["pairs"]:
                    a, b = (u, v) if u < v else (v, u)
                    if a == i:
                        partner, w_ij = b, float(w); break
                    if b == i:
                        partner, w_ij = a, float(w); break
                if partner is not None:
                    r = float(reward_fn(i, partner, jobs))
                    total_savings += r
                    pairs.append((i, partner, float(w_ij), r))
                    available.discard(i); available.discard(partner)
                    paired.update([i, partner])
                    _mark_dispatched((i, partner), t)
                else:
                    solos.append(i)
                    available.discard(i)
                    _mark_dispatched((i,), t)
            if event_hook is not None:
                available_snapshot = tuple(sorted(available))
                due_snapshot = tuple(sorted(i for i in available if due_time[i] <= t))
                event_hook(float(t), available_snapshot, due_snapshot, "after")
            continue

        if policy == "batch2" and is_tick:
            C = set(available)

            def w_eff_periodic(i, j, th):
                w = (weight_fn(i, j, th) if weight_fn is not None else reward_fn(i, j, th))
                if shadow is not None:
                    if i in C:
                        w += float(shadow[i])
                    if j in C:
                        w += float(shadow[j])
                return w

            result = max_weight_matching_subset(list(available), jobs, reward_fn, weight_fn=w_eff_periodic, method="auto")
            matched = set()
            for (i, j, w_weight) in result["pairs"]:
                r = float(reward_fn(i, j, jobs))
                total_savings += r
                pairs.append((i, j, float(w_weight), r))
                matched.update([i, j])
                paired.update([i, j])
                available.discard(i)
                available.discard(j)
                _mark_dispatched((i, j), t)
            for i in sorted(v for v in list(available)):
                solos.append(i)
                available.discard(i)
                _mark_dispatched((i,), t)
            if event_hook is not None:
                available_snapshot = tuple(sorted(available))
                due_snapshot = tuple(sorted(i for i in available if due_time[i] <= t))
                event_hook(float(t), available_snapshot, due_snapshot, "after")
            continue

        if policy == "rbatch2" and is_tick:
            horizon = float(t) + float(tau_s)
            eligible = {i for i in available if due_time[i] <= horizon}
            C = set(eligible)

            def w_eff_periodic(i, j, th):
                w = (weight_fn(i, j, th) if weight_fn is not None else reward_fn(i, j, th))
                if shadow is not None:
                    if i in C:
                        w += float(shadow[i])
                    if j in C:
                        w += float(shadow[j])
                return w

            result = max_weight_matching_subset(list(available), jobs, reward_fn, weight_fn=w_eff_periodic, method="auto")
            dispatched = set()
            for (i, j, w_weight) in result["pairs"]:
                if i not in eligible and j not in eligible:
                    continue
                r = float(reward_fn(i, j, jobs))
                total_savings += r
                pairs.append((i, j, float(w_weight), r))
                dispatched.update([i, j])
                paired.update([i, j])
                available.discard(i)
                available.discard(j)
                _mark_dispatched((i, j), t)
            for i in sorted(eligible - dispatched):
                solos.append(i)
                available.discard(i)
                _mark_dispatched((i,), t)
            if event_hook is not None:
                available_snapshot = tuple(sorted(available))
                due_snapshot = tuple(sorted(i for i in available if due_time[i] <= t))
                event_hook(float(t), available_snapshot, due_snapshot, "after")
            continue

        # SCORE-based greedy paths
        for i in sorted(due_now, key=lambda k: (due_time[k], k)):
            if i not in available:
                continue
            candidates = [j for j in available if j != i]
            if candidates:
                score_map = {cand: float(score_fn(i, cand, jobs)) for cand in candidates}
                best_score = max(score_map.values())
                best_candidates = [
                    cand for cand, val in score_map.items() if np.isclose(val, best_score)
                ]
                if len(best_candidates) == 1:
                    j = best_candidates[0]
                elif tie_breaker == "random":
                    j = int(rng.choice(best_candidates))
                else:  # tie_breaker == "distance"
                    job_i = jobs[i]

                    def _distance_metric(cand: int) -> tuple[float, int]:
                        job_j = jobs[cand]
                        dist = distance(job_i.origin, job_j.origin) + distance(
                            job_i.dest, job_j.dest
                        )
                        return dist, cand

                    j = min(best_candidates, key=_distance_metric)
                score = score_map[j]
                reward = float(reward_fn(i, j, jobs))
            else:
                j, score, reward = None, -1.0, 0.0

            do_match = False
            if j is not None:
                do_match = (score > 0.0) if decision_rule == "threshold" else (reward > 0.0)

            if do_match and j is not None:
                total_savings += reward
                pairs.append((i, j, score, reward))
                available.discard(i); available.discard(j)
                paired.update([i, j])
                _mark_dispatched((i, j), t)
            else:
                solos.append(i)
                available.discard(i)
                _mark_dispatched((i,), t)

        if event_hook is not None:
            available_snapshot = tuple(sorted(available))
            due_snapshot = tuple(sorted(i for i in available if due_time[i] <= t))
            event_hook(float(t), available_snapshot, due_snapshot, "after")

    # leftovers solo
    for i in sorted(list(available)):
        if i not in paired:
            solos.append(i)
            dispatch_time = float(due_time[i]) if np.isfinite(due_time[i]) else float(event_times[-1])
            _mark_dispatched((i,), dispatch_time)

    return {
        "n": n,
        "decision_rule": decision_rule,
        "pairs": pairs,
        "solos": solos,
        "total_savings": float(total_savings),
        "pooled_pct": 100.0 * (2 * len(pairs)) / n if n else 0.0,
        "dispatch_times": dispatch_times,
    }
