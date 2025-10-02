"""Compute H3-based average duals for Meituan delivery datasets.

The pipeline automates the workflow required to train average-dual (AD) tables
from historical Meituan snapshots and then attach those aggregates to a target
day.  It performs three high-level tasks:

1. Load a day's job snapshot, solve the hindsight-dual (HD) linear programme,
   and cache the per-job dual values for reuse.
2. Bucket historical HD samples by the H3 cells of the sender and recipient and
   compute summary statistics for each observed pair.
3. Join the target day's jobs against these aggregates, optionally searching
   neighbouring cells. Unmatched pairs are filled with zeroes so the runtime
   table covers every job. The enriched dataset can then be fed into dispatch
   simulations or exported for analysis.

The script exposes a command-line interface but is also structured so the core
functions can be unit-tested in isolation.
"""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pandas as pd  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore[assignment]

try:
    import h3  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    from ddp._vendor import h3lite as h3  # type: ignore[assignment]

from ddp.engine.opt import compute_lp_relaxation
from ddp.model import Job
from ddp.mappings.h3_pairs import H3PairMapping, make_mapping
from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import reward_fn


# ---------------------------------------------------------------------------
# HD dual generation utilities
# ---------------------------------------------------------------------------


def _load_jobs(path: Path, timestamp_column: str) -> list[Job]:
    jobs = load_jobs_from_csv(path, timestamp_column=timestamp_column)
    if not jobs:
        raise ValueError(f"No jobs were loaded from {path}")
    return jobs


def _solve_hd_duals(jobs: Sequence[Job], deadline: float) -> np.ndarray:
    """Solve the LP relaxation for ``jobs`` and return the dual vector."""

    result = compute_lp_relaxation(jobs, reward_fn, time_window=deadline)
    duals = np.asarray(result["duals"], dtype=float)
    if len(duals) != len(jobs):
        raise RuntimeError("LP solver returned a dual vector with unexpected length")
    return duals


def _require_pandas() -> "pd":
    if pd is None:  # pragma: no cover - defensive
        raise RuntimeError("pandas is required for this operation")
    return pd


def _jobs_to_frame(day: int, jobs: Sequence[Job], duals: Sequence[float]) -> "pd.DataFrame":
    """Convert jobs and duals into a :class:`~pandas.DataFrame`."""

    _pd = _require_pandas()
    rows = []
    for index, (job, dual) in enumerate(zip(jobs, duals)):
        origin_lat, origin_lng = job.origin
        dest_lat, dest_lng = job.dest
        rows.append(
            {
                "day": int(day),
                "job_index": int(index),
                "timestamp": float(job.timestamp),
                "sender_lat": float(origin_lat),
                "sender_lng": float(origin_lng),
                "recipient_lat": float(dest_lat),
                "recipient_lng": float(dest_lng),
                "hindsight_dual": float(dual),
            }
        )
    return _pd.DataFrame.from_records(rows)


def _format_deadline(deadline: float) -> str:
    """Return a stable string representation for ``deadline``."""

    if float(deadline).is_integer():
        return str(int(deadline))
    return format(float(deadline), "g")


def compute_day_hd_duals(
    *,
    day: int,
    snapshot: Path,
    timestamp_column: str,
    deadline: float,
) -> "pd.DataFrame":
    """Load ``snapshot`` and compute the HD duals for ``day``."""

    jobs = _load_jobs(snapshot, timestamp_column=timestamp_column)
    duals = _solve_hd_duals(jobs, deadline=deadline)
    return _jobs_to_frame(day, jobs, duals)


# ---------------------------------------------------------------------------
# H3 bucketing and aggregation
# ---------------------------------------------------------------------------


def add_h3_columns(frame: "pd.DataFrame", mapper: H3PairMapping) -> "pd.DataFrame":
    """Return a copy of ``frame`` with sender/recipient H3 columns added."""

    required = {"sender_lat", "sender_lng", "recipient_lat", "recipient_lng"}
    missing = required.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Frame is missing required columns: {missing_list}")

    _require_pandas()
    sender_hex: list[str] = []
    recipient_hex: list[str] = []
    for row in frame.itertuples(index=False):
        s_hex, r_hex = mapper(
            float(row.sender_lat),
            float(row.sender_lng),
            float(row.recipient_lat),
            float(row.recipient_lng),
        )
        sender_hex.append(str(s_hex))
        recipient_hex.append(str(r_hex))

    result = frame.copy()
    result["sender_hex"] = sender_hex
    result["recipient_hex"] = recipient_hex
    return result


def aggregate_by_hex(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Aggregate ``frame`` by ``(sender_hex, recipient_hex)`` pairs."""

    required = {"sender_hex", "recipient_hex", "hindsight_dual"}
    missing = required.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Frame is missing required columns: {missing_list}")

    grouped = frame.groupby(["sender_hex", "recipient_hex"], sort=True)
    summary = grouped["hindsight_dual"].agg(
        mean_dual="mean",
        std_dual=lambda values: float(np.std(values, ddof=0)),
        count="count",
    )
    summary.reset_index(inplace=True)
    summary["count"] = summary["count"].astype(int)
    summary["type"] = [
        str((sender_hex, recipient_hex))
        for sender_hex, recipient_hex in summary[["sender_hex", "recipient_hex"]].itertuples(index=False, name=None)
    ]
    return summary


# ---------------------------------------------------------------------------
# Average-dual export helpers
# ---------------------------------------------------------------------------


def _ensure_lookup_columns(summary: "pd.DataFrame") -> None:
    required = {"type", "mean_dual"}
    missing = required.difference(summary.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Summary frame is missing required columns: {missing_list}")


def export_average_duals_csv(summary: "pd.DataFrame", output: Path) -> None:
    """Write a runtime-ready CSV lookup table for ``summary``."""

    _require_pandas()
    _ensure_lookup_columns(summary)

    lookup = summary.loc[:, ["type", "mean_dual"]].copy()
    lookup["type"] = lookup["type"].astype(str)
    lookup["mean_dual"] = lookup["mean_dual"].astype(float)

    output.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(output, index=False)
    print(f"Wrote runtime average-dual CSV to {output}")


def export_average_duals_npz(summary: "pd.DataFrame", output: Path) -> None:
    """Write a runtime-ready NPZ lookup table for ``summary``."""

    _ensure_lookup_columns(summary)

    type_series = summary["type"].astype(str)
    if type_series.empty:
        types = np.array([], dtype="<U1")
    else:
        max_len = int(type_series.str.len().max())
        dtype = f"<U{max(1, max_len)}"
        types = np.asarray(type_series.tolist(), dtype=dtype)
    means = summary["mean_dual"].astype(float).to_numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, types=types, mean_dual=means)
    print(f"Wrote runtime average-dual NPZ to {output}")


# ---------------------------------------------------------------------------
# Matching and fallbacks
# ---------------------------------------------------------------------------


def _neighbor_pairs(hex_id: str, radius: int) -> set[str]:
    if radius <= 0:
        return {hex_id}
    if hasattr(h3, "k_ring"):
        return set(h3.k_ring(hex_id, radius))
    if hasattr(h3, "grid_disk"):
        neighbors = h3.grid_disk(hex_id, radius)
        flattened: set[str] = set()
        for entry in neighbors:
            if isinstance(entry, (list, tuple, set)):
                flattened.update(str(cell) for cell in entry)
            else:
                flattened.add(str(entry))
        return flattened
    msg = "Compatible h3 API not available: expected k_ring or grid_disk"
    raise AttributeError(msg)


def match_average_duals(
    target: "pd.DataFrame",
    summary: "pd.DataFrame",
    *,
    neighbor_radius: int = 0,
) -> pd.DataFrame:
    """Attach average-dual estimates to ``target`` using ``summary``.

    Neighbour search is optional; unresolved pairs fall back to zero to ensure
    the runtime lookup covers every target-day job.
    """

    required = {"sender_hex", "recipient_hex", "hindsight_dual"}
    missing = required.difference(target.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Target frame is missing required columns: {missing_list}")

    _require_pandas()
    lookup = {}
    if not summary.empty:
        lookup = {
            (row.sender_hex, row.recipient_hex): float(row.mean_dual)
            for row in summary.itertuples(index=False)
        }

    matched_duals: list[float] = []
    sources: list[str] = []

    for row in target.itertuples(index=False):
        key = (row.sender_hex, row.recipient_hex)
        value = lookup.get(key)
        source = "history" if value is not None else "missing"

        if value is None and neighbor_radius > 0:
            sender_neighbors = _neighbor_pairs(row.sender_hex, neighbor_radius)
            recipient_neighbors = _neighbor_pairs(row.recipient_hex, neighbor_radius)
            for sender_hex in sender_neighbors:
                for recipient_hex in recipient_neighbors:
                    candidate = lookup.get((sender_hex, recipient_hex))
                    if candidate is not None:
                        value = candidate
                        source = "neighbor"
                        break
                if value is not None:
                    break

        if value is None:
            value = 0.0
            source = "zero_fallback"

        matched_duals.append(float(value))
        sources.append(source)

    enriched = target.copy()
    enriched["ad_mean"] = matched_duals
    enriched["ad_source"] = sources
    return enriched


# ---------------------------------------------------------------------------
# Discovery and orchestration helpers
# ---------------------------------------------------------------------------


def _pattern_parts(pattern: str) -> tuple[str, str]:
    if "{day}" not in pattern:
        raise ValueError("jobs_pattern must include '{day}' placeholder")
    prefix, suffix = pattern.split("{day}", 1)
    return prefix, suffix


def discover_available_days(data_dir: Path, jobs_pattern: str) -> list[int]:
    prefix, suffix = _pattern_parts(jobs_pattern)
    wildcard = jobs_pattern.replace("{day}", "*")
    days: list[int] = []
    for path in data_dir.glob(wildcard):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : len(name) - len(suffix) if suffix else None]
        if middle.isdigit():
            days.append(int(middle))
    return sorted(set(days))


def ensure_hd_cache(
    *,
    day: int,
    snapshot: Path,
    cache_dir: Path,
    timestamp_column: str,
    deadline: float,
    force: bool = False,
) -> pd.DataFrame:
    _pd = _require_pandas()
    cache_dir.mkdir(parents=True, exist_ok=True)
    deadline_tag = _format_deadline(deadline)
    cache_path = cache_dir / f"day{day}_d{deadline_tag}_hd.csv"
    if cache_path.exists() and not force:
        return _pd.read_csv(cache_path)

    frame = compute_day_hd_duals(
        day=day,
        snapshot=snapshot,
        timestamp_column=timestamp_column,
        deadline=deadline,
    )
    frame.to_csv(cache_path, index=False)
    return frame


@dataclass
class PipelineResult:
    summary: pd.DataFrame
    lookup: pd.DataFrame


def build_average_duals(
    *,
    day: int,
    data_dir: Path,
    jobs_pattern: str,
    cache_dir: Path,
    timestamp_column: str,
    deadline: float,
    resolution: int,
    history_days: Sequence[int] | None,
    neighbor_radius: int,
    force: bool = False,
) -> PipelineResult:
    _pd = _require_pandas()
    start = time.perf_counter()
    mapper = make_mapping(resolution)

    target_path = data_dir / jobs_pattern.format(day=day)
    if not target_path.exists():
        raise FileNotFoundError(f"Snapshot for day {day} not found: {target_path}")

    target_frame = ensure_hd_cache(
        day=day,
        snapshot=target_path,
        cache_dir=cache_dir,
        timestamp_column=timestamp_column,
        deadline=deadline,
        force=force,
    )
    target_frame = add_h3_columns(target_frame, mapper)

    available_days = discover_available_days(data_dir, jobs_pattern)
    if history_days is None:
        history_days = [d for d in available_days if d != day]
    else:
        history_days = [d for d in history_days if d != day]

    history_frames: list = []
    for history_day in history_days:
        history_path = data_dir / jobs_pattern.format(day=history_day)
        if not history_path.exists():
            continue
        frame = ensure_hd_cache(
            day=history_day,
            snapshot=history_path,
            cache_dir=cache_dir,
            timestamp_column=timestamp_column,
            deadline=deadline,
            force=force,
        )
        frame = add_h3_columns(frame, mapper)
        history_frames.append(frame)

    if history_frames:
        history_frame = _pd.concat(history_frames, ignore_index=True)
        summary = aggregate_by_hex(history_frame)
    else:
        summary = _pd.DataFrame(
            columns=["sender_hex", "recipient_hex", "mean_dual", "std_dual", "count", "type"]
        )

    enriched_target = match_average_duals(
        target_frame,
        summary,
        neighbor_radius=neighbor_radius,
    )

    target_types = [
        str((row.sender_hex, row.recipient_hex))
        for row in enriched_target.itertuples(index=False)
    ]
    lookup = (
        enriched_target.assign(type=target_types)
        .groupby("type", as_index=False)["ad_mean"]
        .mean()
        .rename(columns={"ad_mean": "mean_dual"})
    )

    elapsed = time.perf_counter() - start
    print(
        "Processed day %d with %d history days (resolution=%d) in %.2fs"
        % (day, len(history_days), resolution, elapsed)
    )
    return PipelineResult(summary=summary, lookup=lookup)


def save_summary_map(summary: "pd.DataFrame", output: Path) -> None:
    """Render a Folium map visualising sender hex coverage."""

    if summary.empty:
        print("Summary is empty; skipping map export")
        return

    try:
        import folium
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError("Folium is required for map export") from exc

    counts = summary.groupby("sender_hex")["count"].sum()
    centroids = np.array([h3.h3_to_geo(hex_id) for hex_id in counts.index])
    center = centroids.mean(axis=0)

    fmap = folium.Map(location=center.tolist(), zoom_start=12, control_scale=True)
    max_count = counts.max()
    for hex_id, count in counts.items():
        boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
        opacity = 0.2 + 0.6 * (count / max_count)
        folium.Polygon(
            locations=boundary,
            color="#1f78b4",
            fill=True,
            fill_color="#1f78b4",
            fill_opacity=float(min(0.9, opacity)),
            weight=0.8,
            popup=f"{hex_id}: {count} jobs",
        ).add_to(fmap)

    output.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(output)
    print(f"Saved sender coverage map to {output}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _parse_day_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    days = []
    for item in items:
        days.append(int(item))
    return days


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=int, required=True, help="Target day to process")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Meituan CSV snapshots",
    )
    parser.add_argument(
        "--jobs-pattern",
        default="meituan_city_lunchtime_plat10301330_day{day}.csv",
        help="Filename pattern with '{day}' placeholder for snapshots",
    )
    parser.add_argument(
        "--timestamp-column",
        default="platform_order_time",
        help="Timestamp column name in the CSV files",
    )
    parser.add_argument(
        "--deadline",
        "--d",
        dest="deadline",
        type=float,
        default=20.0,
        help="Slack/timeout parameter passed to the LP relaxation",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=8,
        help="H3 resolution used to bucket sender/recipient coordinates",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=0,
        help="Radius for neighbour search when a hex pair is missing",
    )
    parser.add_argument(
        "--history-days",
        type=_parse_day_list,
        help="Comma-separated list of history days (default: all except target)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hd_cache"),
        help="Directory for cached per-day HD dual CSV files",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/average_duals"),
        help="Directory for default exports when explicit paths are omitted (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute HD duals even when cached files exist",
    )
    parser.add_argument(
        "--export-summary",
        type=Path,
        help="Optional CSV path for the aggregated average-dual table",
    )
    parser.add_argument(
        "--export-ad-csv",
        type=Path,
        help="Optional CSV path for a runtime-ready average-dual lookup",
    )
    parser.add_argument(
        "--folium-map",
        type=Path,
        help="Optional HTML path for a Folium sender coverage map",
    )
    return parser


def _install_signal_handlers() -> None:
    def _handle(signum, _frame) -> None:  # pragma: no cover - signal handler
        raise SystemExit(f"Interrupted by signal {signum}")

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _install_signal_handlers()

    deadline_tag = _format_deadline(float(args.deadline))
    stem = f"meituan_ad_day{int(args.day)}_d{deadline_tag}_res{int(args.resolution)}"
    default_export_dir = Path(args.export_dir)
    default_export_dir.mkdir(parents=True, exist_ok=True)

    result = build_average_duals(
        day=int(args.day),
        data_dir=args.data_dir,
        jobs_pattern=str(args.jobs_pattern),
        cache_dir=args.cache_dir,
        timestamp_column=str(args.timestamp_column),
        deadline=float(args.deadline),
        resolution=int(args.resolution),
        history_days=args.history_days,
        neighbor_radius=int(args.neighbor_radius),
        force=bool(args.force),
    )

    summary_path = args.export_summary or (default_export_dir / f"{stem}_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(summary_path, index=False)
    print(f"Wrote summary table to {summary_path}")

    ad_csv_path = args.export_ad_csv or (default_export_dir / f"{stem}_lookup.csv")
    export_average_duals_csv(result.lookup, ad_csv_path)

    if args.folium_map:
        save_summary_map(result.summary, args.folium_map)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

