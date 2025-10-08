"""Command-line helpers for running SHADOW × DISPATCH experiments on job lists."""

from __future__ import annotations

import csv
import importlib
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Literal

import numpy as np

from ddp.algorithms.potential import potential as potential_vec
from ddp.engine.opt import compute_lp_relaxation, compute_opt
from ddp.engine.sim import simulate
from ddp.model import Job, generate_jobs, reward as pooling_reward
from ddp.scripts.csv_loader import load_jobs_from_csv

_POLICY_DEFAULT_GAMMA: dict[str, float] = {
    "greedy": 1.0,
    "greedy+": 1.0,
    "batch": 0.5,
    "batch+": 1.0,
    "rbatch": 0.5,
    "rbatch+": 1.0,
}


_AD_RESOLUTION_PATTERN = re.compile(r"_res(?P<resolution>[^_]+)_full\Z", re.IGNORECASE)


def _extract_resolution_from_path(path: Path) -> str | None:
    """Return the resolution token extracted from ``path`` if present."""

    match = _AD_RESOLUTION_PATTERN.search(path.stem)
    if match:
        return match.group("resolution")
    return None


def _resolution_sort_key(resolution: str) -> tuple[int, float | str]:
    """Return a deterministic ordering key favouring lower numeric values."""

    match = re.search(r"\d+(?:\.\d+)?", resolution)
    if match:
        try:
            return (0, float(match.group()))
        except ValueError:  # pragma: no cover - defensive, shouldn't happen
            pass
    return (1, resolution)


def _candidate_ad_files(spec: str) -> list[Path]:
    """Return possible AD lookup files for ``spec`` considering globs and prefixes."""

    path = Path(spec).expanduser()
    parent = path.parent if path.parent != Path("") else Path(".")
    name = path.name

    if any(char in name for char in "*?[]"):
        matches = sorted(parent.glob(name))
        return [match for match in matches if match.is_file()]

    if path.is_dir():
        return [match for match in sorted(path.glob("*_res*_full.csv")) if match.is_file()]

    if path.exists():
        return [path]

    pattern = f"{name}*_res*_full.csv" if name else "*_res*_full.csv"
    return [match for match in sorted(parent.glob(pattern)) if match.is_file()]


def _resolve_ad_resolution_map(
    spec: str,
    resolutions: Sequence[str] | None,
) -> dict[str, Path]:
    """Map resolution identifiers to lookup files based on ``spec`` and filters."""

    base_path = Path(spec).expanduser()
    requested = [res.strip() for res in (resolutions or []) if res and res.strip()]
    requested_set = set(requested)
    candidates = _candidate_ad_files(spec)

    if not requested_set and base_path.is_file():
        parent = base_path.parent if base_path.parent != Path("") else Path(".")
        for match in sorted(parent.glob("*_res*_full.csv")):
            if match.is_file() and match not in candidates:
                candidates.append(match)

    mapping: dict[str, Path] = {}
    for candidate in candidates:
        resolution = _extract_resolution_from_path(candidate)
        if resolution is None:
            continue
        mapping.setdefault(resolution, candidate)

    if requested_set:
        missing = sorted(res for res in requested_set if res not in mapping)
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                f"Unable to locate AD resolution(s): {missing_str} under {spec}"
            )
        mapping = {res: mapping[res] for res in requested if res in mapping}

    return mapping


class AverageDualError(RuntimeError):
    """Raised when average-dual shadows cannot be constructed."""


@dataclass(frozen=True)
class AverageDualTable:
    """Container for precomputed average-dual data.

    Attributes
    ----------
    format:
        Discriminator describing the underlying representation. ``"job"``
        indicates an array already aligned to job indices, while ``"type"``
        indicates a sparse lookup keyed by serialized type identifiers.
    by_job:
        NumPy array of per-job values when the source file is job aligned.
    by_type:
        Mapping from serialized type identifiers to average dual values when the
        source file contains aggregated entries.
    """

    format: Literal["job", "type"]
    by_job: np.ndarray | None = None
    by_type: Mapping[str, float] | None = None


def load_average_duals(
    path: str,
    *,
    as_table: bool = True,
) -> AverageDualTable | dict[str, float] | np.ndarray:
    """Load an average-dual lookup table from ``path``.

    Supported formats are ``.npz`` archives containing parallel ``types`` and
    ``mean_dual`` (or ``duals``) arrays, *type-based* CSV/TSV files with
    ``type``/``mean_dual`` (``ad_mean`` or ``dual`` also accepted) columns, and
    *job-aligned* CSV/TSV files that provide one of the job-index headers
    (``job_index`` or ``index``) alongside ``mean_dual``/``ad_mean``/``dual``
    values. Header matching is case-insensitive and ignores surrounding
    whitespace. When ``as_table`` is ``True`` (the default) an
    :class:`AverageDualTable` is returned with either the ``by_job`` or
    ``by_type`` payload populated depending on the input. Setting ``as_table``
    to ``False`` preserves the legacy behaviour of returning the raw payload
    (NumPy array for job-aligned tables, ``dict`` for type-based tables).
    Duplicate entries overwrite previous values. A ``ValueError`` is raised when
    required headers are missing or indices are inconsistent.
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
        table = {key: float(value) for key, value in table.items()}
        if as_table:
            return AverageDualTable(format="type", by_type=table)
        return table

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Average-dual CSV must include a header row")
        normalised: dict[str, str] = {}
        for header in reader.fieldnames:
            stripped = header.strip()
            key = stripped.lower()
            normalised.setdefault(key, header)

        lowered = {key: normalised[key] for key in normalised}
        job_index_key = next(
            (lowered[column] for column in ("job_index", "index") if column in lowered),
            None,
        )
        dual_key = None
        for candidate in ("mean_dual", "ad_mean", "dual"):
            if candidate in lowered:
                dual_key = lowered[candidate]
                break
        if dual_key is None:
            raise ValueError(
                "Average-dual CSV missing 'mean_dual', 'ad_mean', or 'dual' column"
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
                array = np.zeros(0, dtype=float)
                if as_table:
                    return AverageDualTable(format="job", by_job=array)
                return array

            max_index = max(job_values)
            missing = [idx for idx in range(max_index + 1) if idx not in job_values]
            if missing:
                missing_str = ", ".join(str(idx) for idx in missing)
                raise ValueError(
                    "Job-aligned average-dual CSV missing values for indices: "
                    + missing_str
                )

            ordered = [job_values[idx] for idx in range(max_index + 1)]
            array = np.asarray(ordered, dtype=float)
            if as_table:
                return AverageDualTable(format="job", by_job=array)
            return array

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

    if as_table:
        return AverageDualTable(format="type", by_type=table)
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
    table: AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray,
    *,
    mapper: Callable[[Job], str | None] | None = None,
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

    if isinstance(table, AverageDualTable):
        if table.by_job is not None:
            array = np.asarray(table.by_job, dtype=float)
            if array.shape[0] != n:
                raise AverageDualError(
                    "Average-dual array does not match the number of jobs"
                )
            return array

        if table.by_type is not None:
            if mapper is None:
                raise AverageDualError(
                    "Type-indexed average-dual tables require an --ad-mapping function"
                )
            lookup = table.by_type
            shadows = np.empty(n, dtype=float)
            missing_indices: list[int] = []
            missing_types: list[str] = []
            for idx, job in enumerate(jobs):
                type_key = mapper(job)
                if type_key is None:
                    missing_indices.append(idx)
                    missing_types.append("<none>")
                    continue
                str_key = str(type_key)
                value = lookup.get(str_key)
                if value is None:
                    missing_indices.append(idx)
                    missing_types.append(str_key)
                    continue
                shadows[idx] = float(value)

            if missing_indices:
                formatted = ", ".join(
                    f"{idx}:{type_name}" for idx, type_name in zip(missing_indices, missing_types)
                )
                raise AverageDualError(
                    "Missing average-dual values for mapped job types: " + formatted
                )
            return shadows

        raise AverageDualError("Average-dual table did not contain job or type data")

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
        "ad_resolution",
        "n",
        "d",
        "seed",
        "savings",
        "pooled_pct",
        "ratio_lp",
        "lp_gap",
        "ratio_opt",
        "opt_gap",
        "gamma",
        "tau",
        "gamma_plus",
        "tau_plus",
        "pairs",
        "solos",
        "time_s",
        "method",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class _DispatchCandidate:
    """Container for the metrics produced by a dispatch policy."""

    dispatch: str
    row: dict[str, Any]
    run_time: float
    detail: dict[str, Any] | None
    resolution: str | None


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
    gamma: float | None = None,
    tau: float = 0.0,
    gamma_plus: float | None = 1.0,
    tau_plus: float | None = None,
    tie_breaker: str = "distance",
    ad_duals: AverageDualTable
    | Mapping[object, float]
    | Sequence[float]
    | np.ndarray
    | None = None,
    ad_mapper: Callable[[Job], str | None] | None = None,
    ad_duals_by_resolution:
    Mapping[str, Sequence[float] | np.ndarray]
    | Sequence[tuple[str, Sequence[float] | np.ndarray]]
    | None = None,
):
    """Run the SHADOW × DISPATCH grid on a job instance.

    Shadow potentials can be scaled and shifted via ``gamma`` (multiplicative)
    and ``tau`` (additive) before being used by the dispatch policies. The
    effective vector is computed as ``sp = base * gamma + tau`` so the additive
    term raises the scaled shadow. The additive shift applies uniformly to all
    shadow families, including ``"naive"``. When ``gamma`` is omitted (``None``) a
    policy-specific default is used: ``1`` for ``greedy``/``greedy+``, ``0.5``
    for ``batch``/``rbatch``, and ``1`` for the ``+`` variants. The ``+`` variants use the
    :func:`make_weight_fn_latest_shadow` helper, which subtracts only the later
    job's shadow ("late-arrival" adjustment). Their scaling defaults to
    ``gamma_plus = 1`` with optional ``tau_plus`` shifts, and the effective
    weight is ``reward(i, j) - (base * gamma_plus + tau_plus)``. ``tie_breaker``
    selects how greedy policies resolve score ties ("distance" by default, or
    "random" using the provided seed).

    When ``"ad"`` is present in ``shadows`` the optional ``ad_duals`` lookup must
    already contain the runtime shadows aligned with ``jobs``. Provide either a
    job-aligned array/sequence, an :class:`AverageDualTable` with ``by_job`` set,
    or a table keyed by type strings alongside ``ad_mapper`` to translate jobs to
    type identifiers. Missing entries raise :class:`AverageDualError`. When
    multiple precomputed job-aligned arrays are supplied via
    ``ad_duals_by_resolution`` the ``"ad"`` shadow is evaluated for each
    resolution and the dispatch policy results are compared, favouring higher
    savings and then the lowest resolution (numeric when possible) on ties.
    """

    def _format_shadow_label(shadow_name: str, resolution_label: str | None) -> str:
        base = shadow_name.upper()
        if resolution_label:
            return f"{base}({resolution_label})"
        return base

    def _print_result(shadow_name: str, resolution_label: str | None, candidate: _DispatchCandidate) -> None:
        if not print_table:
            return
        row = candidate.row
        pooled_pct = float(row["pooled_pct"])
        savings = float(row["savings"])
        ratio_lp = float(row["ratio_lp"])
        lp_gap = float(row["lp_gap"])
        ratio_opt_val = row.get("ratio_opt")
        opt_gap_val = row.get("opt_gap")
        if with_opt and opt_total is not None and ratio_opt_val is not None and opt_gap_val is not None:
            ratio_opt_str = f"{float(ratio_opt_val):5.2f}x"
            opt_gap_str = f"{float(opt_gap_val):9.3f}"
        else:
            ratio_opt_str = "  n/a"
            opt_gap_str = "     n/a"
        shadow_label = _format_shadow_label(shadow_name, resolution_label)
        print(
            f"{shadow_label:<10} {candidate.dispatch:<12} {pooled_pct:7.1f}% {savings:9.3f}  "
            f"{ratio_lp:5.2f}x {lp_gap:9.3f}  {ratio_opt_str} {opt_gap_str}  "
            f"{row['pairs']:>6}  {row['solos']:>6}  {candidate.run_time:7.3f}"
        )

    def _select_candidate(
        items: list[tuple[str | None, _DispatchCandidate]]
    ) -> tuple[str | None, _DispatchCandidate]:
        if not items:
            raise ValueError("candidate list cannot be empty")

        def sort_key(item: tuple[str | None, _DispatchCandidate]) -> tuple:
            resolution, candidate = item
            savings_val = float(candidate.row.get("savings", float("nan")))
            if math.isnan(savings_val):
                savings_val = float("-inf")
            res_key = (1, "") if resolution is None else _resolution_sort_key(resolution)
            return (-savings_val, res_key[0], res_key[1], resolution or "")

        sorted_items = sorted(items, key=sort_key)
        return sorted_items[0]

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

    rows: list[dict[str, Any]] = []
    details: dict[Any, dict[str, Any]] = {}

    for sh in shadows:
        variant_entries: list[tuple[str | None, np.ndarray]] = []
        if sh == "naive":
            variant_entries.append((None, np.zeros(n, dtype=float)))
        elif sh == "pb":
            variant_entries.append((None, potential_vec(lengths)))
        elif sh == "hd":
            variant_entries.append((None, duals))
        elif sh == "ad":
            if ad_duals_by_resolution is not None:
                if isinstance(ad_duals_by_resolution, Mapping):
                    items = list(ad_duals_by_resolution.items())
                else:
                    items = list(ad_duals_by_resolution)
                items.sort(key=lambda item: _resolution_sort_key(item[0]))
                for resolution_label, payload in items:
                    array = np.asarray(payload, dtype=float)
                    if array.shape[0] != n:
                        raise AverageDualError(
                            "Average-dual array does not match the number of jobs"
                        )
                    variant_entries.append((resolution_label, array))
            else:
                if ad_duals is None:
                    raise AverageDualError(
                        "Average-dual shadows require the precomputed 'ad_duals' table."
                    )
                sp_base = _load_precomputed_ad_shadows(jobs, ad_duals, mapper=ad_mapper)
                variant_entries.append((None, sp_base))
        else:
            if print_table:
                print(f"[skip] Unknown shadow: {sh}")
            continue

        if not variant_entries:
            continue

        variant_results: dict[str | None, dict[str, _DispatchCandidate]] = {}

        for resolution_label, sp_base in variant_entries:
            dispatch_candidates: dict[str, _DispatchCandidate] = {}
            for disp in dispatches:
                t_run = time.perf_counter()
                gamma_value: float | None = None
                tau_value: float | None = None
                gamma_plus_value: float | None = None
                tau_plus_value: float | None = None
                if disp == "greedy":
                    gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[disp]
                    tau_eff = tau
                    sp = np.array(sp_base, dtype=float, copy=True)
                    sp = sp * gamma_eff + tau_eff
                    score_fn = make_local_score(reward_fn, sp)
                    gamma_value = float(gamma_eff)
                    tau_value = float(tau_eff)
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
                    gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[disp]
                    tau_eff = tau
                    sp = np.array(sp_base, dtype=float, copy=True)
                    sp = sp * gamma_eff + tau_eff
                    score_fn = make_local_score(reward_fn, sp)
                    gamma_value = float(gamma_eff)
                    tau_value = float(tau_eff)
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
                    gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[disp]
                    tau_eff = tau
                    sp = np.array(sp_base, dtype=float, copy=True)
                    sp = sp * gamma_eff + tau_eff
                    score_fn = make_local_score(reward_fn, sp)
                    w_fn = make_weight_fn(reward_fn, sp)
                    gamma_value = float(gamma_eff)
                    tau_value = float(tau_eff)
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
                    gamma_plus_eff = gamma_plus if gamma_plus is not None else 1.0
                    tau_plus_eff = tau_plus if tau_plus is not None else 0.0
                    sp_plus = np.array(sp_base, dtype=float, copy=True)
                    sp_plus = sp_plus * gamma_plus_eff + tau_plus_eff
                    score_plus = make_local_score(reward_fn, sp_plus)
                    weight_plus = make_weight_fn_latest_shadow(reward_fn, sp_plus)
                    gamma_plus_value = float(gamma_plus_eff)
                    tau_plus_value = float(tau_plus_eff)
                    res = simulate(
                        jobs,
                        score_plus,
                        reward_fn,
                        "policy",
                        time_window=d,
                        policy="batch",
                        weight_fn=weight_plus,
                        shadow=None,
                        seed=seed,
                        tie_breaker=tie_breaker,
                    )
                elif disp == "rbatch":
                    gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[disp]
                    tau_eff = tau
                    sp = np.array(sp_base, dtype=float, copy=True)
                    sp = sp * gamma_eff + tau_eff
                    score_fn = make_local_score(reward_fn, sp)
                    w_fn = make_weight_fn(reward_fn, sp)
                    gamma_value = float(gamma_eff)
                    tau_value = float(tau_eff)
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
                    gamma_plus_eff = gamma_plus if gamma_plus is not None else 1.0
                    tau_plus_eff = tau_plus if tau_plus is not None else 0.0
                    sp_plus = np.array(sp_base, dtype=float, copy=True)
                    sp_plus = sp_plus * gamma_plus_eff + tau_plus_eff
                    score_plus = make_local_score(reward_fn, sp_plus)
                    weight_plus = make_weight_fn_latest_shadow(reward_fn, sp_plus)
                    gamma_plus_value = float(gamma_plus_eff)
                    tau_plus_value = float(tau_plus_eff)
                    res = simulate(
                        jobs,
                        score_plus,
                        reward_fn,
                        "policy",
                        time_window=d,
                        policy="rbatch",
                        weight_fn=weight_plus,
                        shadow=None,
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
                ratio_opt_val: float | None = None
                gap_opt_val: float | None = None
                if with_opt and opt_total is not None:
                    ratio_opt_val = (r / opt_total) if opt_total > 0 else float("nan")
                    gap_opt_val = _safe_gap(opt_total, r)
                ratio_opt_for_row = (
                    ratio_opt_val if (with_opt and opt_total and opt_total > 0) else None
                )
                opt_gap_for_row = gap_opt_val if (with_opt and opt_total is not None) else None

                row = {
                    "shadow": sh,
                    "dispatch": disp,
                    "ad_resolution": resolution_label if sh == "ad" else None,
                    "n": n,
                    "d": d if np.isscalar(d) else None,
                    "seed": seed,
                    "savings": r,
                    "pooled_pct": pooled_pct,
                    "ratio_lp": ratio_lp,
                    "lp_gap": gap_lp,
                    "ratio_opt": ratio_opt_for_row,
                    "opt_gap": opt_gap_for_row,
                    "pairs": len(res["pairs"]),
                    "solos": len(res["solos"]),
                    "time_s": run_time,
                    "method": ("score" if "greedy" in disp else disp),
                    "gamma": gamma_value,
                    "tau": tau_value,
                    "gamma_plus": gamma_plus_value,
                    "tau_plus": tau_plus_value,
                }

                detail: dict[str, Any] | None = None
                if return_details or print_matches:
                    pairs_idx = [(i, j) for (i, j, *_rest) in res["pairs"]]
                    detail = {"pairs": pairs_idx, "solos": list(res["solos"])}

                dispatch_candidates[disp] = _DispatchCandidate(
                    dispatch=disp,
                    row=row,
                    run_time=run_time,
                    detail=detail,
                    resolution=resolution_label,
                )

            if dispatch_candidates:
                variant_results[resolution_label] = dispatch_candidates

        if not variant_results:
            continue

        if len(variant_results) > 1:
            for resolution_label, dispatch_map in variant_results.items():
                metrics: list[str] = []
                for disp in dispatches:
                    candidate = dispatch_map.get(disp)
                    if candidate is None:
                        continue
                    metrics.append(f"{disp}={candidate.row['savings']:.3f}")
                if metrics:
                    label = resolution_label or "<unspecified>"
                    print(f"[ad] {sh} resolution {label}: " + ", ".join(metrics))

        for disp in dispatches:
            candidate_list = [
                (resolution_label, dispatch_map[disp])
                for resolution_label, dispatch_map in variant_results.items()
                if disp in dispatch_map
            ]
            if not candidate_list:
                continue
            best_resolution, best_candidate = _select_candidate(candidate_list)
            if sh != "ad":
                best_candidate.row["ad_resolution"] = None
            rows.append(best_candidate.row)
            _print_result(sh, best_resolution if sh == "ad" else None, best_candidate)

            if return_details or print_matches:
                detail = best_candidate.detail
                if detail is not None:
                    details[(sh, disp)] = detail
                    if print_matches:
                        suffix = f" (res={best_resolution})" if best_resolution else ""
                        print(
                            f"    -> matches {sh}/{disp}{suffix}: pairs={detail['pairs']}  solos={detail['solos']}"
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
            "gamma": None,
            "tau": None,
            "gamma_plus": None,
            "tau_plus": None,
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
    gamma: float | None = None,
    tau: float = 0.0,
    gamma_plus: float | None = 1.0,
    tau_plus: float | None = None,
    tie_breaker: str = "distance",
    ad_duals: AverageDualTable
    | Mapping[object, float]
    | Sequence[float]
    | np.ndarray
    | None = None,
    ad_mapper: Callable[[Job], str | None] | None = None,
) -> dict:
    """Single-run helper mirroring :func:`run_instance` for one configuration.

    The optional ``gamma``/``tau`` parameters mirror those in :func:`run_instance`
    with the same defaults (policy-specific ``gamma`` and ``tau`` added
    to the scaled shadows, with the additive offset applied to every shadow
    family). For ``dispatch="batch+"`` or
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
        sp_base = _load_precomputed_ad_shadows(jobs, ad_duals, mapper=ad_mapper)
    else:
        raise ValueError(f"Unknown shadow: {shadow}")

    t_run = time.perf_counter()
    if dispatch == "greedy":
        gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[dispatch]
        tau_eff = tau
        sp = np.array(sp_base, dtype=float, copy=True)
        sp = sp * gamma_eff + tau_eff
        score_fn = make_local_score(reward_fn, sp)
        gamma_value = float(gamma_eff)
        tau_value = float(tau_eff)
        gamma_plus_value: float | None = None
        tau_plus_value: float | None = None
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
        gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[dispatch]
        tau_eff = tau
        sp = np.array(sp_base, dtype=float, copy=True)
        sp = sp * gamma_eff + tau_eff
        score_fn = make_local_score(reward_fn, sp)
        gamma_value = float(gamma_eff)
        tau_value = float(tau_eff)
        gamma_plus_value = None
        tau_plus_value = None
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
        gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[dispatch]
        tau_eff = tau
        sp = np.array(sp_base, dtype=float, copy=True)
        sp = sp * gamma_eff + tau_eff
        score_fn = make_local_score(reward_fn, sp)
        w_fn = make_weight_fn(reward_fn, sp)
        gamma_value = float(gamma_eff)
        tau_value = float(tau_eff)
        gamma_plus_value = None
        tau_plus_value = None
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
        gamma_value = None
        tau_value = None
        gamma_plus_value = float(gamma_plus_eff)
        tau_plus_value = float(tau_plus_eff)
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
        gamma_eff = gamma if gamma is not None else _POLICY_DEFAULT_GAMMA[dispatch]
        tau_eff = tau
        sp = np.array(sp_base, dtype=float, copy=True)
        sp = sp * gamma_eff + tau_eff
        score_fn = make_local_score(reward_fn, sp)
        w_fn = make_weight_fn(reward_fn, sp)
        gamma_value = float(gamma_eff)
        tau_value = float(tau_eff)
        gamma_plus_value = None
        tau_plus_value = None
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
        gamma_value = None
        tau_value = None
        gamma_plus_value = float(gamma_plus_eff)
        tau_plus_value = float(tau_plus_eff)
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
        "gamma": gamma_value,
        "tau": tau_value,
        "gamma_plus": gamma_plus_value,
        "tau_plus": tau_plus_value,
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
        default=None,
        help=(
            "Scale factor applied to the shadow potentials before dispatch. "
            "When omitted, uses policy-specific defaults (1 for greedy/greedy+, "
            "0.5 for batch/rbatch, 1 for batch+/rbatch+)."
        ),
    )
    p.add_argument(
        "--tau",
        type=float,
        default=0.0,
        help=(
            "Additive offset subtracted from the scaled shadow potentials before dispatch."
        ),
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
            "Additive offset subtracted from the late-arrival ('+' variants) shadow potentials. "
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
    p.add_argument(
        "--ad-resolution",
        dest="ad_resolution",
        action="append",
        default=None,
        help=(
            "Specific AD resolution to evaluate. Can be repeated; implies job-level CSVs "
            "named *_res*_full.csv."
        ),
    )
    p.add_argument(
        "--ad-resolutions",
        dest="ad_resolutions",
        default=None,
        help=(
            "Comma-separated list of AD resolutions to evaluate. Blank or omitted triggers auto discovery."
        ),
    )
    p.add_argument(
        "--ad-mapping",
        help=(
            "Module:function resolving to an average-dual mapper used with type-indexed tables."
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

    ad_resolution_inputs: list[str] = []
    seen_resolutions: set[str] = set()
    if args.ad_resolution:
        for value in args.ad_resolution:
            if not value:
                continue
            cleaned = value.strip()
            if cleaned and cleaned not in seen_resolutions:
                ad_resolution_inputs.append(cleaned)
                seen_resolutions.add(cleaned)
    if args.ad_resolutions is not None:
        for part in args.ad_resolutions.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in seen_resolutions:
                ad_resolution_inputs.append(cleaned)
                seen_resolutions.add(cleaned)

    ad_table: AverageDualTable | Mapping[object, float] | Sequence[float] | np.ndarray | None = None
    ad_by_resolution: dict[str, np.ndarray] | None = None
    ad_mapper: Callable[[Job], str | None] | None = None
    if args.ad_mapping:
        try:
            ad_mapper = load_average_dual_mapper(args.ad_mapping)
        except (ModuleNotFoundError, AttributeError, ValueError, TypeError) as exc:
            raise SystemExit(f"failed to resolve --ad-mapping: {exc}") from exc
    if "ad" in shadow_list:
        if not args.ad_duals:
            raise SystemExit("--shadows contains 'ad' so --ad_duals is required")
        try:
            resolution_map = _resolve_ad_resolution_map(args.ad_duals, ad_resolution_inputs)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

        if resolution_map:
            ad_by_resolution = {}
            for resolution, path in sorted(
                resolution_map.items(), key=lambda item: _resolution_sort_key(item[0])
            ):
                try:
                    table = load_average_duals(str(path))
                except (OSError, ValueError) as exc:
                    raise SystemExit(f"failed to load AD lookup {path}: {exc}") from exc
                try:
                    ad_values = _load_precomputed_ad_shadows(jobs, table, mapper=ad_mapper)
                except AverageDualError as exc:
                    raise SystemExit(str(exc)) from exc
                ad_by_resolution[resolution] = ad_values
        else:
            try:
                ad_table = load_average_duals(args.ad_duals)
            except (OSError, ValueError) as exc:
                raise SystemExit(f"failed to load --ad-duals: {exc}") from exc
            if (
                isinstance(ad_table, AverageDualTable)
                and ad_table.by_job is None
                and ad_table.by_type is not None
                and ad_mapper is None
            ):
                raise SystemExit("type-indexed average-dual tables require --ad-mapping")
    elif ad_mapper is not None:
        raise SystemExit("--ad-mapping can only be used when --shadows contains 'ad'")

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
        ad_mapper=ad_mapper,
        ad_duals_by_resolution=ad_by_resolution,
    )


if __name__ == "__main__":
    main()
