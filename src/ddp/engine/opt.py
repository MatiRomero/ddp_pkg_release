from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ddp.model import Job

# ---------------------------- helpers ----------------------------
def _normalize_deadlines(n: int, time_window) -> np.ndarray | None:
    """Normalize deadline specification to an array aligned with ``jobs``."""

    if time_window is None:
        return None

    if np.isscalar(time_window):
        return np.full(n, float(time_window), dtype=float)

    deadlines = np.asarray(time_window, dtype=float)
    if deadlines.shape != (n,):
        raise ValueError("time_window array must have length equal to len(jobs)")
    return deadlines


def _build_positive_edges(jobs: Sequence[Job], weight_fn, time_window=None):
    """Build feasible edges ``(i, j, w)`` with positive weight."""
    n = len(jobs)
    edges: List[Tuple[int, int, float]] = []
    deadlines = _normalize_deadlines(n, time_window)
    if deadlines is not None:
        timestamps = np.array([float(job.timestamp) for job in jobs], dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if deadlines is not None:
                t_i = timestamps[i]
                t_j = timestamps[j]
                latest_start = max(t_i, t_j)
                earliest_deadline = min(t_i + deadlines[i], t_j + deadlines[j])
                if latest_start > earliest_deadline:
                    continue
            w = float(weight_fn(i, j, jobs))
            if w > 0:
                edges.append((i, j, w))
    return edges

# ---------------------------- LP (upper bound) + duals ----------------------------
def compute_lp_relaxation(jobs: Sequence[Job], reward_fn, time_window=None):
    """Fractional matching LP (upper bound) + duals (hindsight shadow prices).

    ``time_window`` mirrors :func:`compute_opt` and restricts feasible edges to
    those whose availability windows overlap.
    """
    try:
        import pulp
    except Exception as e:
        raise RuntimeError("LP requires 'pulp'. `pip install pulp`") from e

    jobs = list(jobs)
    n = len(jobs)

    def w_ij(i, j, th):
        return reward_fn(i, j, th)

    edges = _build_positive_edges(jobs, w_ij, time_window=time_window)
    prob = pulp.LpProblem("fractional_matching", pulp.LpMaximize)
    x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0.0, 1.0, cat="Continuous") for (i, j, _) in edges}
    prob += pulp.lpSum(w * x[(i, j)] for (i, j, w) in edges)
    for v in range(n):
        prob += pulp.lpSum(x[(i, j)] for (i, j) in x if i == v or j == v) <= 1.0
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    total_upper = float(pulp.value(prob.objective) or 0.0)
    frac = []
    for (i, j), var in x.items():
        val = float(var.value() or 0.0)
        if val > 1e-9:
            w = next(w for (ii, jj, w) in edges if ii == i and jj == j)
            frac.append((i, j, val, float(w)))
    frac.sort()

    # dual
    dual = pulp.LpProblem("fractional_matching_dual", pulp.LpMinimize)
    y = [pulp.LpVariable(f"y_{i}", lowBound=0.0, cat="Continuous") for i in range(n)]
    dual += pulp.lpSum(y)
    for (i, j, w) in edges:
        dual += y[i] + y[j] >= w
    dual.solve(pulp.PULP_CBC_CMD(msg=False))
    duals = [float(var.value() or 0.0) for var in y]

    return {"total_upper": total_upper, "frac_edges": frac, "duals": duals, "method": "ilp"}

# ---------------------------- Offline OPT (integral) ----------------------------
def compute_opt(jobs: Sequence[Job], reward_fn, method: str = "auto", time_window=None):
    """Offline OPT as integral max-weight matching on the reward graph.

    ``time_window`` may be ``None`` (all jobs mutually compatible), a scalar
    deadline applied to every job, or an array-like of per-job deadlines.  In
    the scalar case an edge ``(i, j)`` is retained only when ``|t_i - t_j|`` is
    at most the deadline.  For array inputs an edge is feasible when the
    availability intervals ``[t_i, t_i + d_i]`` and ``[t_j, t_j + d_j]``
    overlap.
    """
    jobs = list(jobs)

    def w_ij(i, j, th):
        return reward_fn(i, j, th)

    edges = _build_positive_edges(jobs, w_ij, time_window=time_window)
    n = len(jobs)

    if method in ("auto", "networkx"):
        try:
            import networkx as nx
            G = nx.Graph(); G.add_nodes_from(range(n))
            for i, j, w in edges: G.add_edge(i, j, weight=w)
            M = nx.algorithms.matching.max_weight_matching(G, weight="weight")
            pairs, total = [], 0.0
            for u, v in M:
                i, j = (u, v) if u < v else (v, u)
                w = float(G[i][j]["weight"])
                pairs.append((i, j, w)); total += w
            pairs.sort()
            return {"total_reward": float(total), "pairs": pairs, "method": "networkx"}
        except Exception:
            if method == "networkx": raise

    if method in ("auto", "ilp"):
        try:
            import pulp
            prob = pulp.LpProblem("max_weight_matching", pulp.LpMaximize)
            x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat="Binary") for (i, j, _) in edges}
            prob += pulp.lpSum(w * x[(i, j)] for (i, j, w) in edges)
            for v in range(n):
                prob += pulp.lpSum(x[(i, j)] for (i, j) in x if i == v or j == v) <= 1
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
            pairs, total = [], 0.0
            wmap = {(i, j): w for (i, j, w) in edges}
            for (i, j), var in x.items():
                if (var.value() or 0) > 0.5:
                    w = float(wmap[(i, j)])
                    pairs.append((i, j, w)); total += w
            pairs.sort()
            return {"total_reward": float(total), "pairs": pairs, "method": "ilp"}
        except Exception:
            if method == "ilp": raise

    raise RuntimeError("No OPT method available. Install 'networkx' or 'pulp'.")

# ---------------------------- Subset matching (batch / rbatch) ----------------------------
def max_weight_matching_subset(nodes, jobs: Sequence[Job], reward_fn, weight_fn=None, method: str = "auto"):
    """Max-weight matching on subset S with edge weight_fn (defaults to reward)."""
    if weight_fn is None:
        weight_fn = reward_fn
    S = sorted(set(int(v) for v in nodes))
    if not S:
        return {"total_weight": 0.0, "pairs": [], "method": "none"}

    jobs = list(jobs)

    def w_ij(i, j, th):
        return weight_fn(i, j, th)
    edges = []
    for a in range(len(S)):
        i = S[a]
        for b in range(a + 1, len(S)):
            j = S[b]
            w = float(w_ij(i, j, jobs))
            if w > 0:
                edges.append((i, j, w))

    if method in ("auto", "networkx"):
        try:
            import networkx as nx
            G = nx.Graph(); G.add_nodes_from(S)
            for i, j, w in edges: G.add_edge(i, j, weight=w)
            M = nx.algorithms.matching.max_weight_matching(G, weight="weight")
            pairs, total = [], 0.0
            for u, v in M:
                i, j = (u, v) if u < v else (v, u)
                w = float(G[i][j]["weight"])
                pairs.append((i, j, w)); total += w
            pairs.sort()
            return {"total_weight": float(total), "pairs": pairs, "method": "networkx"}
        except Exception:
            if method == "networkx": raise

    if method in ("auto", "ilp"):
        try:
            import pulp
            prob = pulp.LpProblem("subset_max_weight_matching", pulp.LpMaximize)
            x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat="Binary") for (i, j, _) in edges}
            prob += pulp.lpSum(w * x[(i, j)] for (i, j, w) in edges)
            for v in S:
                prob += pulp.lpSum(x[(i, j)] for (i, j) in x if i == v or j == v) <= 1
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            pairs, total = [], 0.0
            wmap = {(i, j): w for (i, j, w) in edges}
            for (i, j), var in x.items():
                if (var.value() or 0) > 0.5:
                    w = float(wmap[(i, j)])
                    pairs.append((i, j, w)); total += w
            pairs.sort()
            return {"total_weight": float(total), "pairs": pairs, "method": "ilp"}
        except Exception:
            if method == "ilp": raise

    raise RuntimeError("No matching method available. Install 'networkx' or 'pulp'.")
