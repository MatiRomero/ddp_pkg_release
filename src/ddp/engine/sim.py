from typing import Sequence

import numpy as np

from ddp.engine.opt import max_weight_matching_subset
from ddp.model import Job


def simulate(
    jobs: Sequence[Job],
    score_fn,           # local score(i,j,theta) = reward - s_j
    reward_fn,          # reward(i,j,theta)
    decision_rule="naive",         # 'naive' or 'threshold' (ignored for batch/rbatch)
    timestamps=None,               # array or None → 1,2,3,...
    time_window=None,              # scalar or array
    policy="score",                # 'score' | 'batch' | 'rbatch'
    weight_fn=None,                # base weight for matching: reward - s_i - s_j
    shadow=None,                   # vector s_i (needed for critical adjustment)
    seed=0,
):
    jobs = list(jobs)
    n = len(jobs)

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
    event_times = np.unique(np.concatenate([timestamps, due_time]))
    arrived = np.zeros(n, dtype=bool)
    available = set()
    paired = set()
    total_savings = 0.0
    pairs = []  # (i, j, score_or_weight, reward)
    solos = []

    for t in event_times:
        # arrivals
        to_add = np.where((~arrived) & (timestamps <= t))[0]
        for i in to_add:
            available.add(int(i))
            arrived[i] = True

        # due now
        due_now = [i for i in available if due_time[i] <= t]
        if not due_now:
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
            for i in sorted(v for v in available if v not in matched):
                solos.append(i)
            available.clear()
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
                else:
                    solos.append(i)
                    available.discard(i)
            continue

        # SCORE-based greedy paths
        for i in sorted(due_now, key=lambda k: (due_time[k], k)):
            if i not in available:
                continue
            candidates = [j for j in available if j != i]
            if candidates:
                j = max(candidates, key=lambda j_: score_fn(i, j_, jobs))
                score = float(score_fn(i, j, jobs))
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
            else:
                solos.append(i)
                available.discard(i)

    # leftovers solo
    for i in sorted(list(available)):
        if i not in paired:
            solos.append(i)

    return {
        "n": n,
        "decision_rule": decision_rule,
        "pairs": pairs,
        "solos": solos,
        "total_savings": float(total_savings),
        "pooled_pct": 100.0 * (2 * len(pairs)) / n if n else 0.0,
    }
