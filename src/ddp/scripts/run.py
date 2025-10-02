"""Command-line helpers for running SHADOW × DISPATCH experiments on job lists."""

from __future__ import annotations

import csv
import importlib
import os
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ddp.algorithms.potential import potential as potential_vec
from ddp.engine.opt import compute_lp_relaxation, compute_opt
from ddp.engine.sim import simulate
from ddp.model import Job, generate_jobs, reward as pooling_reward
from ddp.scripts.csv_loader import load_jobs_from_csv

DispatchState = tuple[
    Callable[[int, int, Sequence[Job]], float],
    Callable[[int, int, Sequence[Job]], float],
    Optional[np.ndarray],
]


class AverageDualError(RuntimeError):
    """Raised when average-dual shadows cannot be constructed."""


def load_average_duals(path: str) -> dict[str, float] | np.ndarray:
    """Load an average-dual lookup table from ``path``.

    Supported formats are ``.npz`` archives containing parallel ``types`` and
    ``mean_dual`` (or ``duals``) arrays, *type-based* CSV/TSV files with
    ``type``/``mean_dual`` (or ``dual``) columns, and *job-aligned* CSV/TSV files
    that provide one of the job-index headers (``job_index`` or ``index``)
    alongside ``mean_dual``/``dual`` values. Type-based files return a mapping
    keyed by the serialized type, while job-aligned files return a NumPy array
    ordered by job index. Duplicate entries overwrite previous values. A
    ``ValueError`` is raised when required headers are missing or indices are
    inconsistent.
    """

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        table: dict[str, float] = {}
        with np.load(path, allow_pickle=False) as data:
            if "types" not in data:
                raise ValueError("Average-dual archive must contain a 'types' array")
            if "mean_dual" in data:
                means = data["mean_dual"]
            elif "duals" in data:
                means = data["duals"]
            else:
                raise ValueError("Average-dual archive must contain 'mean_dual' or 'duals'")
            types = data["types"]
            if len(types) != len(means):
                raise ValueError("Mismatch between 'types' and dual arrays in archive")
            for key, value in zip(types, means):
                table[str(key)] = float(value)
        return table

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Average-dual CSV must include a header row")
        lowered = {name.lower(): name for name in reader.fieldnames}
        job_index_key = next(
            (lowered[column] for column in ("job_index", "index") if column in lowered),
            None,
        )
        dual_key = lowered.get("mean_dual", lowered.get("dual"))
        if dual_key is None:
            raise ValueError(
                "Average-dual CSV missing 'mean_dual' (or 'dual') column"
            )

        if job_index_key is not None:
            job_values: dict[int, float] = {}
            for row in reader:
                if not row:
                    continue
                raw_index = row[job_index_key].strip()
                if not raw_index:
                    continue
                try:
                    job_idx = int(raw_index)
                except ValueError as exc:
                    raise ValueError(
                        "Job-aligned average-dual CSV must contain integer job indices"
                    ) from exc
                if job_idx < 0:
                    raise ValueError("Job indices must be non-negative")
                try:
                    value = float(row[dual_key])
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                    raise ValueError(
                        f"Invalid dual value for job index {job_idx}"
                    ) from exc
                job_values[job_idx] = value

            if not job_values:
                return np.zeros(0, dtype=float)

            max_index = max(job_values)
            missing = [idx for idx in range(max_index + 1) if idx not in job_values]
            if missing:
                missing_str = ", ".join(str(idx) for idx in missing)
                raise ValueError(
                    "Job-aligned average-dual CSV missing values for indices: "
                    + missing_str
                )

            ordered = [job_values[idx] for idx in range(max_index + 1)]
            return np.asarray(ordered, dtype=float)

        if "type" not in lowered:
            raise ValueError(
                "Average-dual CSV must include a 'type' column or one of the job index "
                "columns: 'job_index', 'index'"
            )

        table: dict[str, float] = {}
        type_key = lowered["type"]
        for row in reader:
            if not row:
                continue
            type_name = row[type_key].strip()
            if not type_name:
                continue
            try:
                value = float(row[dual_key])
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid dual value for type '{type_name}'") from exc
            table[type_name] = value
    return table


def load_average_dual_mapper(spec: str) -> Callable[[Job], str | None]:
    """Resolve ``module:function`` import paths to average-dual mappers."""

    if ":" not in spec:
        raise ValueError("Mapping spec must be 'module:function'")
    module_name, func_name = spec.rsplit(":", 1)
    if not module_name or not func_name:
        raise ValueError("Mapping spec must include both module and function name")
    module = importlib.import_module(module_name)
    try:
        func = getattr(module, func_name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Module '{module_name}' lacks attribute '{func_name}'") from exc
    if not callable(func):
        raise TypeError(f"Resolved object '{module_name}:{func_name}' is not callable")
    return func


def _load_precomputed_ad_shadows(
    jobs: Sequence[Job],
    table: Mapping[object, float] | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the precomputed average-dual shadows for ``jobs``.

    Runtime AD evaluation now expects the lookup to be *job aligned*: the
    provided ``table`` must already contain one value per job either as a
    positional sequence/array or as an explicit mapping keyed by the job index
    (``0``-based). Any missing entry triggers :class:`AverageDualError`.
    """

    n = len(jobs)
    if n == 0:
        return np.zeros(0, dtype=float)

    if isinstance(table, np.ndarray):
        if table.shape[0] != n:
            raise AverageDualError(
                "Average-dual array does not match the number of jobs"
            )
        return np.asarray(table, dtype=float)

    if isinstance(table, Sequence) and not isinstance(table, (str, bytes)):
        array = np.asarray(list(table), dtype=float)
        if array.shape[0] != n:
            raise AverageDualError(
                "Average-dual sequence does not match the number of jobs"
            )
        return array

    if isinstance(table, Mapping):
        shadows = np.empty(n, dtype=float)
        missing: list[int] = []
        for idx in range(n):
            value: float | None = None
            if idx in table:
                value = float(table[idx])  # type: ignore[index]
            else:
                key = str(idx)
                raw = table.get(key)
                if raw is not None:
                    value = float(raw)
            if value is None:
                missing.append(idx)
            else:
                shadows[idx] = value
        if missing:
            missing_str = ", ".join(str(idx) for idx in missing)
            raise AverageDualError(
                "Missing average-dual values for job indices: " + missing_str
            )
        return shadows

    raise AverageDualError(
        "Average-dual table must be an array, sequence, or mapping indexed by job"
    )


def reward_fn(i: int, j: int, jobs: Sequence[Job]) -> float:
    """Toy pooling reward: distance saved when merging jobs ``i`` and ``j``."""

    return pooling_reward([jobs[i], jobs[j]])


def make_local_score(reward_fn, sp):
    """Local greedy score: ``r(i, j) - s_j``."""

    def score(i: int, j: int, jobs: Sequence[Job]) -> float:
        return reward_fn(i, j, jobs) - float(sp[j])

    return score


def make_weight_fn(reward_fn, sp):
    """(R)BATCH weight: ``r(i, j) - s_i - s_j``."""

    def weight(i: int, j: int, jobs: Sequence[Job]) -> float:
        return reward_fn(i, j, jobs) - float(sp[i]) - float(sp[j])

    return weight


def make_weight_fn_latest_shadow(reward_fn, sp):
    """Critical-aware weight using only the later job's shadow value."""

    def weight(i: int, j: int, jobs: Sequence[Job]) -> float:
        job_i = jobs[i]
        job_j = jobs[j]
        if (job_i.timestamp > job_j.timestamp) or (
            job_i.timestamp == job_j.timestamp and i > j
        ):
            later_idx = i
        else:
            later_idx = j
        return reward_fn(i, j, jobs) - float(sp[later_idx])

    return weight


def _safe_gap(upper: float, val: float) -> float:
    return max(upper - val, 0.0)


def _write_csv(rows, path: str) -> None:
    if not path:
        return
    fields = [
        "shadow",
        "dispatch",
        "n",
        "d",
        "seed",
        "savings",
        "pooled_pct",
        "ratio_lp",
        "lp_gap",
        "ratio_opt",
        "opt_gap",
        "pairs",
        "solos",
        "time_s",
        "method",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_instance(
    jobs: Sequence[Job],
    d,
    shadows=("naive", "pb", "hd"),
    dispatches=("greedy", "greedy+", "batch", "batch+", "rbatch", "rbatch+"),
    seed=0,
    with_opt=False,
    opt_method="auto",
    save_csv="",
    print_table=True,
    return_details=False,
    print_matches=False,
    gamma: float = 0.5,
    tau: float = 0.0,
    gamma_plus: float | None = 1.0,
    tau_plus: float | None = None,
    tie_breaker: str = "distance",
    ad_duals: Mapping[object, float] | Sequence[float] | np.ndarray | None = None,
):
    """Run the SHADOW × DISPATCH grid on a job instance.

    Shadow potentials can be scaled and shifted via ``gamma`` (multiplicative)
    and ``tau`` (additive) before being used by the dispatch policies. ``gamma``
    defaults to 0.5 so the standard BATCH/RBATCH heuristics subtract both job
    shadows with the same scaling used in prior releases. The ``+`` variants use
    the :func:`make_weight_fn_latest_shadow` helper, which subtracts only the
    later job's shadow ("late-arrival" adjustment). Their scaling defaults to
    ``gamma_plus = 1`` with optional ``tau_plus`` shifts, and the effective
    weight is strictly ``reward(i, j) - s_late``. ``tie_breaker`` selects how
    greedy policies resolve score ties ("distance" by default, or "random"
    using the provided seed).

    When ``"ad"`` is present in ``shadows`` the optional ``ad_duals`` lookup must
    already contain the runtime shadows aligned with ``jobs``. Provide either a
    sequence/NumPy array whose length matches ``len(jobs)`` or a mapping keyed by
    the ``0``-based job index. Missing entries raise :class:`AverageDualError`.
    """

    jobs = list(jobs)
    n = len(jobs)
    timestamps = np.array([job.timestamp for job in jobs], dtype=float)
    lengths = np.array([job.length for job in jobs], dtype=float)

    # compute due_time for plotting/debug (not used by simulate)
    if np.isscalar(d):
        due_time = timestamps + float(d)
    else:
        d_arr = np.asarray(d, dtype=float)
        assert len(d_arr) == n, "len(time_window) must equal len(jobs)"
        due_time = timestamps + d_arr

    # LP (upper bound + duals for HD) — compute once
    t0 = time.perf_counter()
    lp = compute_lp_relaxation(jobs, reward_fn, time_window=d)
    lp_time = time.perf_counter() - t0
    lp_total = float(lp["total_upper"])
    duals = np.array(lp["duals"], dtype=float)

    # Optional OPT once
    opt_total = opt_pairs = opt_m = None
    opt_time = 0.0
    if with_opt:
        t0 = time.perf_counter()
        opt = compute_opt(jobs, reward_fn, method=opt_method, time_window=d)
        opt_time = time.perf_counter() - t0
        opt_total = float(opt["total_reward"])
        opt_pairs = opt["pairs"]
        opt_m = opt["method"]

    if print_table:
        print(f"LP_RELAX  upper={lp_total:.3f}  method={lp['method']}  time={lp_time:.3f}s")
        if with_opt:
            print(
                f"OPT       total={opt_total:.3f}  pairs={len(opt_pairs):>4}  method={opt_m}  time={opt_time:.3f}s"
            )
        print(f"\nInstance: n={n}, d={d}")
        print("-" * 142)
        print(
            "SHADOW     DISPATCH     POOLED%   SAVINGS    R/LP   LP_GAP    R/OPT  OPT_GAP   #PAIRS  #SOLOS   TIME(s)"
        )

    rows = []
    details = {}

    for sh in shadows:
        if sh == "naive":
            sp_base = np.zeros(n, dtype=float)
        elif sh == "pb":
            sp_base = potential_vec(lengths)
        elif sh == "hd":
            sp_base = duals
        elif sh == "ad":
            if ad_duals is None:
                raise AverageDualError(
                    "Average-dual shadows require the precomputed 'ad_duals' table."
                )
            sp_base = _load_precomputed_ad_shadows(jobs, ad_duals)
        else:
            if print_table:
                print(f"[skip] Unknown shadow: {sh}")
            continue

        sp = np.array(sp_base, dtype=float, copy=True)
        sp = sp * gamma + tau

        score_fn = make_local_score(reward_fn, sp)
        w_fn = make_weight_fn(reward_fn, sp)

        extra_dispatch_state: dict[str, DispatchState] = {}
        plus_variants = {"batch+", "rbatch+"}
        if any(label in dispatches for label in plus_variants):
            gamma_plus_eff = gamma_plus if gamma_plus is not None else 1.0
            tau_plus_eff = tau_plus if tau_plus is not None else 0.0
            sp_plus = np.array(sp_base, dtype=float, copy=True)
            sp_plus = sp_plus * gamma_plus_eff + tau_plus_eff
            score_plus = make_local_score(reward_fn, sp_plus)
            weight_plus = make_weight_fn_latest_shadow(reward_fn, sp_plus)
            for label in plus_variants:
                if label in dispatches:
                    extra_dispatch_state[label] = (
                        score_plus,
                        weight_plus,
                        None,
                    )

        for disp in dispatches:
            t_run = time.perf_counter()
            if disp == "greedy":
                res = simulate(
                    jobs,
                    score_fn,
                    reward_fn,
                    "naive",
                    time_window=d,
                    policy="score",
                    weight_fn=None,
                    shadow=None,
                    seed=seed,
                    tie_breaker=tie_breaker,
                )
            elif disp == "greedy+":
                res = simulate(
                    jobs,
                    score_fn,
                    reward_fn,
                    "threshold",
                    time_window=d,
                    policy="score",
                    weight_fn=None,
                    shadow=None,
                    seed=seed,
                    tie_breaker=tie_breaker,
                )
            elif disp == "batch":
                res = simulate(
                    jobs,
                    score_fn,
                    reward_fn,
                    "policy",
                    time_window=d,
                    policy="batch",
                    weight_fn=w_fn,
                    shadow=sp,
                    seed=seed,
                    tie_breaker=tie_breaker,
                )
            elif disp == "batch+":
                score_plus, weight_plus, shadow_plus = extra_dispatch_state["batch+"]
                res = simulate(
                    jobs,
                    score_plus,
                    reward_fn,
                    "policy",
                    time_window=d,
                    policy="batch",
                    weight_fn=weight_plus,
                    shadow=shadow_plus,
                    seed=seed,
                    tie_breaker=tie_breaker,
                )
            elif disp == "rbatch":
                res = simulate(
                    jobs,
                    score_fn,
                    reward_fn,
                    "policy",
                    time_window=d,
                    policy="rbatch",
                    weight_fn=w_fn,
                    shadow=sp,
                    seed=seed,
                    tie_breaker=tie_breaker,
                )
            elif disp == "rbatch+":
                score_plus, weight_plus, shadow_plus = extra_dispatch_state["rbatch+"]
                res = simulate(
                    jobs,
                    score_plus,
                    reward_fn,
                    "policy",
                    time_window=d,
                    policy="rbatch",
                    weight_fn=weight_plus,
                    shadow=shadow_plus,
                    seed=seed,
                    tie_breaker=tie_breaker,
                )
            else:
                if print_table:
                    print(f"[skip] Unknown dispatch: {disp}")
                continue
            run_time = time.perf_counter() - t_run

            r = res["total_savings"]
            pooled_pct = res["pooled_pct"]
            ratio_lp = (r / lp_total) if lp_total > 0 else float("nan")
            gap_lp = _safe_gap(lp_total, r)
            if with_opt and opt_total is not None:
                ratio_opt = (r / opt_total) if opt_total > 0 else float("nan")
                gap_opt = _safe_gap(opt_total, r)
                ratio_opt_str = f"{ratio_opt:5.2f}x"
                gap_opt_str = f"{gap_opt:9.3f}"
            else:
                ratio_opt_str = "  n/a"
                gap_opt_str = "     n/a"

            if print_table:
                print(
                    f"{sh.upper():<10} {disp:<12} {pooled_pct:7.1f}% {r:9.3f}  "
                    f"{ratio_lp:5.2f}x {gap_lp:9.3f}  {ratio_opt_str} {gap_opt_str}  "
                    f"{len(res['pairs']):>6}  {len(res['solos']):>6}  {run_time:7.3f}"
                )

            rows.append(
                {
                    "shadow": sh,
                    "dispatch": disp,
                    "n": n,
                    "d": d if np.isscalar(d) else None,
                    "seed": seed,
                    "savings": r,
                    "pooled_pct": pooled_pct,
                    "ratio_lp": ratio_lp,
                    "lp_gap": gap_lp,
                    "ratio_opt": (r / opt_total) if (with_opt and opt_total and opt_total > 0) else None,
                    "opt_gap": (_safe_gap(opt_total, r) if (with_opt and opt_total is not None) else None),
                    "pairs": len(res["pairs"]),
                    "solos": len(res["solos"]),
                    "time_s": run_time,
                    "method": ("score" if "greedy" in disp else disp),
                }
            )

            if return_details or print_matches:
                pairs_idx = [(i, j) for (i, j, _, _) in res["pairs"]]
                info = {"pairs": pairs_idx, "solos": list(res["solos"])}
                details[(sh, disp)] = info
                if print_matches:
                    print(
                        f"    -> matches {sh}/{disp}: pairs={pairs_idx}  solos={info['solos']}"
                    )

    if with_opt and opt_total is not None:
        opt_pairs_list = list(opt_pairs) if opt_pairs is not None else []
        opt_pair_indices = [(i, j) for (i, j, *_rest) in opt_pairs_list]
        matched = {idx for pair in opt_pair_indices for idx in pair}
        opt_solo_indices = sorted(set(range(n)) - matched)
        opt_pair_count = len(opt_pair_indices)
        opt_pooled_pct = (200.0 * opt_pair_count / n) if n > 0 else 0.0
        opt_ratio_lp = (opt_total / lp_total) if lp_total > 0 else float("nan")
        opt_lp_gap = _safe_gap(lp_total, opt_total)

        if print_table:
            ratio_lp_str = f"{opt_ratio_lp:5.2f}x" if lp_total > 0 else "  n/a"
            print(
                f"OPT       {'opt':<12} {opt_pooled_pct:7.1f}% {opt_total:9.3f}  "
                f"{ratio_lp_str} {opt_lp_gap:9.3f}  {1.00:5.2f}x {0.0:9.3f}  "
                f"{opt_pair_count:>6}  {len(opt_solo_indices):>6}  {opt_time:7.3f}"
            )

        opt_row = {
            "shadow": "opt",
            "dispatch": "opt",
            "n": n,
            "d": d if np.isscalar(d) else None,
            "seed": seed,
            "savings": opt_total,
            "pooled_pct": opt_pooled_pct,
            "ratio_lp": opt_ratio_lp,
            "lp_gap": opt_lp_gap,
            "ratio_opt": 1.0,
            "opt_gap": 0.0,
            "pairs": opt_pair_count,
            "solos": len(opt_solo_indices),
            "time_s": opt_time,
            "method": opt_m,
        }
        rows.append(opt_row)

        if return_details or print_matches:
            opt_detail = {"pairs": opt_pair_indices, "solos": opt_solo_indices}
            details["opt"] = opt_detail
            if print_matches:
                print(
                    f"    -> matches OPT/opt: pairs={opt_pair_indices}  solos={opt_solo_indices}"
                )

    _write_csv(rows, save_csv)

    out = {
        "rows": rows,
        "lp_total": lp_total,
        "lp_time": lp_time,
        "opt_total": opt_total,
        "opt_time": opt_time,
        "opt_pairs": opt_pairs,
        "opt_method": opt_m,
        "jobs": jobs,
        "timestamps": timestamps,
        "due_time": due_time,
    }
    if return_details:
        out["details"] = details
    return out


def run_once(
    n: int,
    d: float,
    seed: int,
    shadow: str,
    dispatch: str,
    with_opt: bool = False,
    opt_method: str = "auto",
    gamma: float = 0.5,
    tau: float = 0.0,
    gamma_plus: float | None = 1.0,
    tau_plus: float | None = None,
    tie_breaker: str = "distance",
    ad_duals: Mapping[object, float] | Sequence[float] | np.ndarray | None = None,
) -> dict:
    """Single-run helper mirroring :func:`run_instance` for one configuration.

    The optional ``gamma``/``tau`` parameters mirror those in :func:`run_instance`
    with the same defaults (0.5 and 0). For ``dispatch="batch+"`` or
    ``dispatch="rbatch+"`` the weight computation subtracts only the later job's
    shadow, producing ``reward(i, j) - s_late``. ``gamma_plus`` defaults to 1 (and
    can be overridden) while ``tau_plus`` still shifts the late-arrival shadows.
    ``tie_breaker`` mirrors the option in :func:`run_instance` for resolving greedy
    score ties.
    The ``ad_duals`` lookup, when provided with ``shadow='ad'``, must already
    contain one value per generated job just like :func:`run_instance`.
    """

    rng = np.random.default_rng(seed)
    if n <= 1:
        raise ValueError("run_once requires n > 1 to generate jobs")
    jobs = generate_jobs(n, rng)
    lengths = np.array([job.length for job in jobs], dtype=float)

    lp = compute_lp_relaxation(jobs, reward_fn, time_window=d)
    lp_total = float(lp["total_upper"])
    duals = np.array(lp["duals"], dtype=float)

    opt_total = None
    if with_opt:
        opt = compute_opt(jobs, reward_fn, method=opt_method, time_window=d)
        opt_total = float(opt["total_reward"])

    if shadow == "naive":
        sp_base = np.zeros(n, dtype=float)
    elif shadow == "pb":
        sp_base = potential_vec(lengths)
    elif shadow == "hd":
        sp_base = duals
    elif shadow == "ad":
        if ad_duals is None:
            raise AverageDualError("Average-dual shadows require the 'ad_duals' table")
        sp_base = _load_precomputed_ad_shadows(jobs, ad_duals)
    else:
        raise ValueError(f"Unknown shadow: {shadow}")

    sp = np.array(sp_base, dtype=float, copy=True)
    sp = sp * gamma + tau

    score_fn = make_local_score(reward_fn, sp)
    w_fn = make_weight_fn(reward_fn, sp)

    t_run = time.perf_counter()
    if dispatch == "greedy":
        res = simulate(
            jobs,
            score_fn,
            reward_fn,
            "naive",
            time_window=d,
            policy="score",
            weight_fn=None,
            shadow=None,
            seed=seed,
            tie_breaker=tie_breaker,
        )
    elif dispatch == "greedy+":
        res = simulate(
            jobs,
            score_fn,
            reward_fn,
            "threshold",
            time_window=d,
            policy="score",
            weight_fn=None,
            shadow=None,
            seed=seed,
            tie_breaker=tie_breaker,
        )
    elif dispatch == "batch":
        res = simulate(
            jobs,
            score_fn,
            reward_fn,
            "policy",
            time_window=d,
            policy="batch",
            weight_fn=w_fn,
            shadow=sp,
            seed=seed,
            tie_breaker=tie_breaker,
        )
    elif dispatch == "batch+":
        gamma_plus_eff = gamma_plus if gamma_plus is not None else 1.0
        tau_plus_eff = tau_plus if tau_plus is not None else 0.0
        sp_plus = np.array(sp_base, dtype=float, copy=True)
        sp_plus = sp_plus * gamma_plus_eff + tau_plus_eff
        score_fn_plus = make_local_score(reward_fn, sp_plus)
        w_fn_plus = make_weight_fn_latest_shadow(reward_fn, sp_plus)
        res = simulate(
            jobs,
            score_fn_plus,
            reward_fn,
            "policy",
            time_window=d,
            policy="batch",
            weight_fn=w_fn_plus,
            shadow=None,
            seed=seed,
            tie_breaker=tie_breaker,
        )
    elif dispatch == "rbatch":
        res = simulate(
            jobs,
            score_fn,
            reward_fn,
            "policy",
            time_window=d,
            policy="rbatch",
            weight_fn=w_fn,
            shadow=sp,
            seed=seed,
            tie_breaker=tie_breaker,
        )
    elif dispatch == "rbatch+":
        gamma_plus_eff = gamma_plus if gamma_plus is not None else 1.0
        tau_plus_eff = tau_plus if tau_plus is not None else 0.0
        sp_plus = np.array(sp_base, dtype=float, copy=True)
        sp_plus = sp_plus * gamma_plus_eff + tau_plus_eff
        score_fn_plus = make_local_score(reward_fn, sp_plus)
        w_fn_plus = make_weight_fn_latest_shadow(reward_fn, sp_plus)
        res = simulate(
            jobs,
            score_fn_plus,
            reward_fn,
            "policy",
            time_window=d,
            policy="rbatch",
            weight_fn=w_fn_plus,
            shadow=None,
            seed=seed,
            tie_breaker=tie_breaker,
        )
    else:
        raise ValueError(f"Unknown dispatch: {dispatch}")
    run_time = time.perf_counter() - t_run

    r = res["total_savings"]
    pooled_pct = res["pooled_pct"]
    ratio_lp = (r / lp_total) if lp_total > 0 else float("nan")
    gap_lp = _safe_gap(lp_total, r)
    ratio_opt = (r / opt_total) if (with_opt and opt_total and opt_total > 0) else None
    gap_opt = (_safe_gap(opt_total, r) if (with_opt and opt_total is not None) else None)

    return {
        "shadow": shadow,
        "dispatch": dispatch,
        "n": n,
        "d": d,
        "seed": seed,
        "savings": r,
        "pooled_pct": pooled_pct,
        "ratio_lp": ratio_lp,
        "lp_gap": gap_lp,
        "ratio_opt": ratio_opt,
        "opt_gap": gap_opt,
        "pairs": len(res["pairs"]),
        "solos": len(res["solos"]),
        "time_s": run_time,
        "method": ("score" if "greedy" in dispatch else dispatch),
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run SHADOW×DISPATCH on a given job instance.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--jobs",
        type=str,
        help="Path to .npz containing 'origins', 'dests', 'timestamps'.",
    )
    group.add_argument(
        "--jobs-csv",
        type=str,
        help=(
            "Path to a CSV file that can be parsed by ddp.scripts.csv_loader.load_jobs_from_csv."
        ),
    )
    p.add_argument(
        "--timestamp-column",
        default="platform_order_time",
        help="CSV column containing ISO timestamps (used with --jobs-csv).",
    )
    p.add_argument(
        "--export-npz",
        default=None,
        help="Optional path to write the loaded jobs as an .npz archive.",
    )
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shadows", default="naive,pb,hd")
    p.add_argument(
        "--dispatch",
        default="greedy,greedy+,batch,batch+,rbatch,rbatch+",
        help=(
            "Comma-separated dispatch policies. "
            "The '+ variants apply late-arrival shadow weighting with weights "
            "reward(i, j) - s_late (subtracting only the later job's shadow)."
        ),
    )
    p.add_argument(
        "--with_opt",
        action="store_true",
        help="Compute the OPT baseline with the same deadline parameter 'd'",
    )
    p.add_argument("--opt_method", default="auto", choices=["auto", "networkx", "ilp"])
    p.add_argument("--save_csv", default="")
    p.add_argument("--print_matches", action="store_true")
    p.add_argument("--return_details", action="store_true")
    p.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help=(
            "Scale factor applied to the shadow potentials before dispatch. "
            "Defaults to 0.5 so BATCH/RBATCH match prior releases."
        ),
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help="Additive offset applied to the shadow potentials before dispatch.",
    )
    p.add_argument(
        "--plus_gamma",
        type=float,
        default=1.0,
        help=(
            "Scale factor for the late-arrival ('+' variants) shadow potentials. "
            "Defaults to 1 so the weight is reward(i, j) - s_late unless overridden."
        ),
    )
    p.add_argument(
        "--plus_tau",
        type=float,
        default=None,
        help=(
            "Additive offset for the late-arrival ('+' variants) shadow potentials. "
            "Defaults to 0 when omitted."
        ),
    )
    p.add_argument(
        "--tie_breaker",
        default="distance",
        choices=["distance", "random"],
        help=(
            "Tie-breaking rule for greedy candidate selection when scores are equal. "
            "'distance' prefers the job closest to the critical job; 'random' samples "
            "uniformly using the provided seed."
        ),
    )
    p.add_argument(
        "--ad_duals",
        help=(
            "Path to an average-dual table (.npz or CSV). Required when --shadows includes 'ad'."
        ),
    )
    args = p.parse_args()

    origins: np.ndarray
    dests: np.ndarray
    timestamps: np.ndarray

    if args.jobs:
        with np.load(args.jobs) as data:
            try:
                origins = np.array(data["origins"], dtype=float)
                dests = np.array(data["dests"], dtype=float)
                timestamps = np.array(data["timestamps"], dtype=float)
            except KeyError as exc:
                msg = "--jobs file must contain 'origins', 'dests', and 'timestamps'"
                raise SystemExit(msg) from exc
    elif args.jobs_csv:
        jobs = load_jobs_from_csv(
            args.jobs_csv,
            timestamp_column=args.timestamp_column,
        )
        origins = np.array([job.origin for job in jobs], dtype=float)
        dests = np.array([job.dest for job in jobs], dtype=float)
        timestamps = np.array([job.timestamp for job in jobs], dtype=float)
    else:  # pragma: no cover - argparse enforces exclusivity, but defensive
        raise SystemExit("Provide either --jobs or --jobs-csv")

    if not (len(origins) == len(dests) == len(timestamps)):
        raise SystemExit("Mismatched job array lengths in provided job data")

    jobs = [
        Job(
            origin=tuple(map(float, origin)),
            dest=tuple(map(float, dest)),
            timestamp=float(ts),
        )
        for origin, dest, ts in zip(origins, dests, timestamps)
    ]

    if args.export_npz:
        export_origins = np.array([job.origin for job in jobs], dtype=float)
        export_dests = np.array([job.dest for job in jobs], dtype=float)
        export_timestamps = np.array([job.timestamp for job in jobs], dtype=float)
        np.savez(
            args.export_npz,
            origins=export_origins,
            dests=export_dests,
            timestamps=export_timestamps,
        )

    shadow_list = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    dispatch_list = [d.strip() for d in args.dispatch.split(",") if d.strip()]

    ad_table = load_average_duals(args.ad_duals) if args.ad_duals else None
    if "ad" in shadow_list:
        if ad_table is None:
            raise SystemExit("--shadows contains 'ad' so --ad_duals is required")

    run_instance(
        jobs=jobs,
        d=args.d,
        shadows=shadow_list,
        dispatches=dispatch_list,
        seed=args.seed,
        with_opt=args.with_opt,
        opt_method=args.opt_method,
        save_csv=args.save_csv,
        print_table=True,
        return_details=args.return_details,
        print_matches=args.print_matches,
        gamma=args.gamma,
        tau=args.tau,
        gamma_plus=args.plus_gamma,
        tau_plus=args.plus_tau,
        tie_breaker=args.tie_breaker,
        ad_duals=ad_table,
    )


if __name__ == "__main__":
    main()
