"""Reusable PB-RBAT preparation; bounds are opt-in and computed once per day."""
from __future__ import annotations
from functools import lru_cache
import time
import math

import numpy as np

from ddp.area_gamma import resolve_area_gammas
from ddp.engine.sim import simulate
from ddp.scripts.run import make_local_score, make_weight_fn, make_reward_fn
from ddp.engine.opt import compute_lp_relaxation, compute_opt


class PreparedPB:
    def __init__(self, jobs, d=60, *, with_lp=False, with_opt=False, dispatch="rbatch", tau_s=30):
        start = time.perf_counter()
        self.jobs = list(jobs)
        if dispatch not in {"rbatch", "rbatch2"}:
            raise ValueError("PreparedPB supports rbatch and rbatch2")
        if not math.isfinite(float(d)) or d <= 0:
            raise ValueError("Waiting window must be finite and positive")
        if dispatch == "rbatch2" and (not math.isfinite(float(tau_s)) or not 0 < tau_s <= d):
            raise ValueError("Periodic interval must lie in (0, window]")
        self.dispatch = dispatch
        self.tau_s = tau_s
        self.d = d
        self.lengths = np.asarray([job.length for job in self.jobs])
        self.base = self.lengths / 2
        reward = make_reward_fn('pooling')

        @lru_cache(maxsize=250_000)
        def cached(i, j):
            return reward(i, j, self.jobs)

        self.reward = lambda i, j, jobs: cached(min(i, j), max(i, j))
        self.lp = compute_lp_relaxation(self.jobs, self.reward, time_window=d)['total_upper'] if with_lp else None
        self.opt = compute_opt(self.jobs, self.reward, time_window=d)['total_reward'] if with_opt else None
        self.preparation_s = time.perf_counter() - start

    def run(self, *, gamma=None, gamma_table=None, seed=0):
        gammas, provenance = resolve_area_gammas(self.jobs, gamma_table, d=self.d, gamma=gamma, dispatches=(self.dispatch,), tau_s=self.tau_s)
        effective = gammas if gammas is not None else (0.5 if gamma is None else gamma)
        shadows = self.base * effective
        start = time.perf_counter()
        result = simulate(self.jobs, make_local_score(self.reward, shadows), self.reward,
                          'policy', time_window=self.d, policy=self.dispatch, tau_s=self.tau_s,
                          weight_fn=make_weight_fn(self.reward, shadows), shadow=shadows, seed=seed)
        elapsed = time.perf_counter() - start
        n = len(self.jobs)
        waits = [result['dispatch_times'][i] - j.timestamp for i, j in enumerate(self.jobs)]
        if 2 * len(result['pairs']) + len(result['solos']) != n or len(waits) != n:
            raise ValueError('Incomplete job dispatch accounting')
        if any(w < -1e-9 or w > self.d + 1e-9 for w in waits):
            raise ValueError('Dispatch outside job availability window')
        cross = sum(self.jobs[i].da_id != self.jobs[j].da_id for i, j, *_ in result['pairs'])
        row = {'shadow': 'pb', 'dispatch': self.dispatch, 'tau': 0, 'd': self.d, 'seed': seed,
               'gamma': gamma if gammas is None else None, 'n': n,
               'savings': result['total_savings'], 'pairs': len(result['pairs']),
               'solos': len(result['solos']), 'pooled_pct': result['pooled_pct'],
               'direct_distance_total': float(self.lengths.sum()),
               'savings_fraction': result['total_savings'] / self.lengths.sum() if self.lengths.sum() else 0,
               'mean_wait_seconds': float(np.mean(waits)) if waits else 0,
               'max_wait_seconds': max(waits, default=0),
               'cross_area_pairs': cross, 'cross_area_pair_fraction': cross / len(result['pairs']) if result['pairs'] else 0,
               'time_s': elapsed, 'lp_total': self.lp, 'opt_total': self.opt,
               'ratio_lp': result['total_savings'] / self.lp if self.lp else None,
               'ratio_opt': result['total_savings'] / self.opt if self.opt else None,
               **provenance}
        if self.dispatch == 'rbatch2':
            row['tau_s'] = self.tau_s
            row['tick_origin'] = 'loaded_timestamp_zero'
            if any(not np.isclose(t / self.tau_s, round(t / self.tau_s)) for t in result['dispatch_times'].values()):
                raise ValueError('Periodic dispatch outside a tick')
        return row, result
