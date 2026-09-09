"""Reproduce preliminary Meituan area/H3 diagnostics without running policies.

Uses raw accepted records only to recover area labels for the existing snapshots.
Outputs aggregates, not an enriched copy of the individual order records.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


COORDS = ["sender_lat", "sender_lng", "recipient_lat", "recipient_lng"]
TIMES = ["platform_order_time", "order_push_time"]
KEYS = COORDS + ["dt"] + TIMES
H3_CODE = """
import json, sys, h3
points = json.load(sys.stdin)
fn = getattr(h3, 'latlng_to_cell', None) or h3.geo_to_h3
json.dump([[fn(lat, lng, r) for r in (7, 8, 9)] for lat, lng in points], sys.stdout)
"""


def load_labeled_snapshots(data_dir: Path, raw_path: Path):
    raw = pd.read_csv(raw_path, usecols=KEYS + ["da_id", "is_courier_grabbed"])
    accepted = raw.loc[raw["is_courier_grabbed"].eq(True)].copy()
    for col in TIMES:
        accepted[col] = (
            pd.to_datetime(accepted[col], unit="s") + pd.Timedelta(hours=8)
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    labels = accepted.groupby(KEYS, dropna=False).agg(
        da_id=("da_id", "first"),
        distinct_areas=("da_id", "nunique"),
        raw_multiplicity=("da_id", "size"),
    ).reset_index()
    frames, audit = [], []
    for day in range(8):
        path = data_dir / f"meituan_city_lunchtime_plat10301330_day{day}.csv"
        snap = pd.read_csv(path, usecols=KEYS)
        if snap[KEYS].isna().any().any():
            raise ValueError(f"Missing snapshot join fields: {path}")
        for col in COORDS:
            snap[col] = np.rint(snap[col] * 1_000_000).astype("int64")
        merged = snap.merge(labels, on=KEYS, how="left", validate="many_to_one")
        if merged.da_id.isna().any() or merged.distinct_areas.ne(1).any():
            raise ValueError(f"Missing or ambiguous area labels: {path}")
        audit.append({
            "day": day, "date": int(snap.dt.iloc[0]), "jobs": len(snap),
            "matched_jobs": int(merged.da_id.notna().sum()),
            "ambiguous_area_jobs": int(merged.distinct_areas.ne(1).sum()),
            "multiple_raw_matches_same_area": int(merged.raw_multiplicity.gt(1).sum()),
        })
        merged["day"] = day
        merged["da_id"] = merged.da_id.astype(int)
        for col in COORDS:
            merged[col] = merged[col] / 1_000_000
        ts = pd.to_datetime(merged.platform_order_time)
        merged["time_seconds"] = (ts - ts.dt.normalize()).dt.total_seconds()
        if not merged.time_seconds.between(10.5 * 3600, 13.5 * 3600, inclusive="left").all():
            raise ValueError("Snapshot outside the declared 10:30–13:30 interval")
        frames.append(merged)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(audit), len(raw), len(accepted)


def add_cells(frame: pd.DataFrame, h3_python: str | None):
    origins = frame[["sender_lat", "sender_lng"]].to_numpy()
    destinations = frame[["recipient_lat", "recipient_lng"]].to_numpy()
    points, inverse = np.unique(np.vstack([origins, destinations]), axis=0, return_inverse=True)
    if h3_python:
        completed = subprocess.run(
            [h3_python, "-c", H3_CODE], input=json.dumps(points.tolist()),
            text=True, capture_output=True, check=True,
        )
        values = np.asarray(json.loads(completed.stdout))
    else:
        if importlib.util.find_spec("h3") is None:
            raise RuntimeError("Install real H3 or pass --h3-python with an H3-enabled Python")
        import h3
        fn = getattr(h3, "latlng_to_cell", None) or h3.geo_to_h3
        values = np.asarray([[fn(lat, lng, res) for res in (7, 8, 9)] for lat, lng in points])
    n = len(frame)
    for column, resolution in enumerate((7, 8, 9)):
        frame[f"origin_h3_{resolution}"] = values[inverse[:n], column]
        frame[f"destination_h3_{resolution}"] = values[inverse[n:], column]


def area_purity(frame: pd.DataFrame, cell_column: str):
    """Fraction agreeing with the modal area of their endpoint's H3 cell."""
    counts = frame.groupby([cell_column, "da_id"]).size()
    return float(counts.groupby(level=0).max().sum() / len(frame))


def endpoint_area_consistency(frame: pd.DataFrame, output: Path):
    rows = []
    for endpoint in ("sender", "recipient"):
        counts = frame.groupby([f"{endpoint}_lat", f"{endpoint}_lng", "da_id"]).size()
        grouped = counts.groupby(level=[0, 1])
        totals, distinct = grouped.sum(), grouped.size()
        rows.append({
            "endpoint": endpoint, "unique_recorded_locations": len(totals),
            "locations_with_multiple_area_ids": int(distinct.gt(1).sum()),
            "fraction_jobs_at_multiple_area_locations": float(totals[distinct.gt(1)].sum() / len(frame)),
            "modal_area_agreement": float(grouped.max().sum() / len(frame)),
        })
    pd.DataFrame(rows).to_csv(output / "endpoint_area_consistency.csv", index=False)


def summarize_zones(frame: pd.DataFrame, output: Path):
    summaries, daily_tables = [], {}
    for label, origin, destination in [
        ("Area ID", "da_id", None),
        *[(f"H3 res {r}", f"origin_h3_{r}", f"destination_h3_{r}") for r in (7, 8, 9)],
    ]:
        daily = pd.crosstab(frame[origin], frame.day).reindex(columns=range(8), fill_value=0)
        totals = daily.sum(axis=1)
        history = totals.to_numpy()[:, None] - daily.to_numpy()
        daily_tables[label] = daily
        row = {
            "zoning": label, "origin_or_assigned_zones": len(daily),
            "destination_zones": None, "observed_od_pairs": None,
            "median_jobs_per_zone_per_day": float((totals / 8).median()),
            "median_seven_day_history_jobs": float(np.median(history)),
            "min_total_jobs": int(totals.min()), "max_total_jobs": int(totals.max()),
            "zones_observed_all_days": int(daily.gt(0).all(axis=1).sum()),
            "cross_od_fraction": None, "cross_od_day_min": None, "cross_od_day_max": None,
            "origin_h3_modal_area_agreement": None, "destination_h3_modal_area_agreement": None,
        }
        if destination:
            cross = frame[origin].ne(frame[destination])
            cross_day = cross.groupby(frame.day).mean()
            od = frame.groupby([origin, destination]).size().rename("jobs").reset_index()
            od.to_csv(output / f"od_counts_{origin}.csv", index=False)
            row.update({
                "destination_zones": int(frame[destination].nunique()),
                "observed_od_pairs": len(od), "cross_od_fraction": float(cross.mean()),
                "cross_od_day_min": float(cross_day.min()), "cross_od_day_max": float(cross_day.max()),
                "origin_h3_modal_area_agreement": area_purity(frame, origin),
                "destination_h3_modal_area_agreement": area_purity(frame, destination),
            })
            overlap = frame.groupby(["da_id", origin]).size().rename("jobs").reset_index()
            overlap.to_csv(output / f"area_overlap_{origin}.csv", index=False)
        daily.to_csv(output / f"daily_counts_{origin}.csv")
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "zone_summary.csv", index=False)
    return summary, daily_tables


def pair_rewards(origins, destinations, i, j):
    """Same Euclidean all-pickups-before-dropoffs reward as ddp.model.reward."""
    def dist(a, b):
        return np.linalg.norm(a - b, axis=1)
    links = np.minimum.reduce([
        dist(origins[i], destinations[i]), dist(origins[i], destinations[j]),
        dist(origins[j], destinations[i]), dist(origins[j], destinations[j]),
    ])
    return (dist(origins[i], destinations[i]) + dist(origins[j], destinations[j])
            - dist(origins[i], origins[j]) - dist(destinations[i], destinations[j]) - links)


def sample_potential_pairs(frame, window, sample_size, seed, output):
    ordered = frame.sort_values(["day", "time_seconds"], kind="stable").reset_index(drop=True)
    lower = np.zeros(len(ordered), dtype=int)
    for _, group in ordered.groupby("day", sort=False):
        start = int(group.index[0])
        times = group.time_seconds.to_numpy()
        lower[group.index] = start + np.searchsorted(times, times - window, side="left")
    eligible_counts = np.arange(len(ordered)) - lower
    cumulative = np.cumsum(eligible_counts)
    population = int(cumulative[-1])
    if population == 0:
        raise ValueError("No temporally eligible pairs")
    rng = np.random.default_rng(seed)
    ranks = rng.integers(population, size=sample_size)
    i = np.searchsorted(cumulative, ranks, side="right")
    previous = np.where(i > 0, cumulative[np.maximum(i - 1, 0)], 0)
    j = lower[i] + ranks - previous
    assert np.all(j < i)
    assert np.all(ordered.day.to_numpy()[i] == ordered.day.to_numpy()[j])
    assert np.all(ordered.time_seconds.to_numpy()[i] - ordered.time_seconds.to_numpy()[j] <= window)
    origins = ordered[["sender_lat", "sender_lng"]].to_numpy()
    destinations = ordered[["recipient_lat", "recipient_lng"]].to_numpy()
    rewards = pair_rewards(origins, destinations, i, j)
    # Independently enumerate the four routes for a small sample.
    for a, b, value in zip(i[:20], j[:20], rewards[:20]):
        direct = np.linalg.norm(origins[a] - destinations[a]) + np.linalg.norm(origins[b] - destinations[b])
        routes = []
        for first, second in ((a, b), (b, a)):
            for third, fourth in ((a, b), (b, a)):
                routes.append(np.linalg.norm(origins[first] - origins[second])
                              + np.linalg.norm(origins[second] - destinations[third])
                              + np.linalg.norm(destinations[third] - destinations[fourth]))
        assert np.isclose(value, direct - min(routes), atol=1e-12)
    positive = rewards > 1e-12
    rows = []
    for label, col in [("Area ID", "da_id"), *[(f"H3 res {r}", f"origin_h3_{r}") for r in (7, 8, 9)]]:
        labels = ordered[col].to_numpy()
        cross = labels[i] != labels[j]
        rows.append({
            "zoning": label, "window_seconds": window, "seed": seed,
            "eligible_pair_population": population, "sampled_pairs": sample_size,
            "positive_reward_pairs": int(positive.sum()),
            "cross_fraction_all_sampled_pairs": float(cross.mean()),
            "cross_fraction_positive_reward_pairs": float(cross[positive].mean()),
            "cross_share_sum_positive_rewards": float(rewards[positive & cross].sum() / rewards[positive].sum()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "potential_pair_sample.csv", index=False)
    return result


def plot_summary(summary, daily_tables, pairs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    colors = ["#283593", "#00897b", "#e49a20", "#b43e62"]
    for (label, table), color in zip(daily_tables.items(), colors):
        counts = np.sort(table.sum(axis=1).to_numpy() / 8)
        axes[0].step(counts, np.arange(1, len(counts) + 1) / len(counts), where="post", label=label, color=color)
    axes[0].set(xscale="log", xlabel="Jobs per zone per day (eight-day mean)", ylabel="Fraction of zones", title="Data available per coefficient")
    axes[0].legend(frameon=False)
    h3_rows = summary.iloc[1:]
    bars = axes[1].bar(h3_rows.zoning, 100 * h3_rows.cross_od_fraction, color=colors[1:])
    axes[1].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[1].set(ylim=(0, 105), ylabel="Orders (%)", title="Destination outside origin cell")
    bars = axes[2].bar(pairs.zoning, 100 * pairs.cross_fraction_positive_reward_pairs, color=colors)
    axes[2].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[2].set(ylim=(0, 105), ylabel="Positive-reward candidate pairs (%)", title="Two jobs assigned to different zones")
    axes[2].tick_params(axis="x", labelrotation=15)
    fig.suptitle("Meituan zoning diagnostics · eight lunch snapshots", fontsize=15)
    fig.supxlabel("Candidate pairs: uniform sample of 100,000 pairs arriving within 60 seconds; potential matches, not policy outcomes.", fontsize=9)
    fig.savefig(output / "zone_diagnostics.png", dpi=180)
    fig.savefig(output / "zone_diagnostics.pdf")
    plt.close(fig)


def plot_saved_summary(output):
    summary = pd.read_csv(output / "zone_summary.csv")
    pairs = pd.read_csv(output / "potential_pair_sample.csv")
    daily_tables = {
        label: pd.read_csv(output / f"daily_counts_{column}.csv", index_col=0)
        for label, column in [("Area ID", "da_id"), *[(f"H3 res {r}", f"origin_h3_{r}") for r in (7, 8, 9)]]
    }
    plot_summary(summary, daily_tables, pairs, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw", type=Path, default=Path("../data/all_waybill_info_meituan_0322.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/zone_diagnostics"))
    parser.add_argument("--h3-python", help="Optional Python executable for real H3 conversion")
    parser.add_argument("--plot-python", help="Optional Python executable with matplotlib")
    parser.add_argument("--plot-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_saved_summary(output)
        return
    frame, audit, raw_rows, accepted_rows = load_labeled_snapshots(args.data_dir, args.raw)
    audit.to_csv(output / "join_audit.csv", index=False)
    endpoint_area_consistency(frame, output)
    print(f"Recovered unambiguous areas for {len(frame):,} snapshot jobs", flush=True)
    add_cells(frame, args.h3_python)
    summary, daily_tables = summarize_zones(frame, output)
    pairs = sample_potential_pairs(frame, 60, 100_000, 20260908, output)
    if args.plot_python:
        subprocess.run([args.plot_python, str(Path(__file__).resolve()), "--plot-only", "--output-dir", str(output)], check=True)
    else:
        plot_summary(summary, daily_tables, pairs, output)
    provenance = {
        "raw_source": str(args.raw.resolve()), "snapshot_directory": str(args.data_dir.resolve()),
        "raw_rows": raw_rows, "accepted_raw_rows": accepted_rows,
        "snapshot_jobs": len(frame), "days": 8, "observation_seconds_per_day": 10800,
        "area_join": "accepted raw records; exact microdegree OD coordinates, dt, and both timestamps",
        "raw_timestamp_conversion": "Unix seconds + 8 hours, matching the existing notebook",
        "area_od_crossing": "unavailable: one da_id per order, no endpoint area labels or boundaries",
        "pair_sample": "uniform with replacement over all unordered same-day pairs with arrival gap <= 60 seconds",
        "pair_reward": "current simulator Euclidean coordinate reward; all pickups precede deliveries",
        "pair_sample_seed": 20260908, "pair_sample_size": 100000,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_json(orient="records", indent=2))
    print(pairs.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
