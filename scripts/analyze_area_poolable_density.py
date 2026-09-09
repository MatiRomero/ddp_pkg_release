"""Compare each job's future pooling opportunities with historical area density.

Counts all pairs exactly, without policy dispatches. A poolable future arrival
has positive pooling reward and timestamp in (arrival, arrival + window].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from analyze_meituan_zones import load_labeled_snapshots, pair_rewards


EXPOSURE_SECONDS = 10800
OBSERVATION_END = 13.5 * 3600


def count_future_opportunities(day_frame, window, area_ids, chunk_size=250_000):
    ordered = day_frame.sort_values("time_seconds", kind="stable").reset_index(drop=True)
    times = ordered.time_seconds.to_numpy()
    origins = ordered[["sender_lat", "sender_lng"]].to_numpy()
    destinations = ordered[["recipient_lat", "recipient_lng"]].to_numpy()
    area = pd.Categorical(ordered.da_id, categories=area_ids).codes.astype(int)
    n, n_areas = len(ordered), len(area_ids)
    if np.any(area < 0):
        raise ValueError("Unknown area")
    start = np.searchsorted(times, times, side="right")
    end = np.searchsorted(times, times + window, side="right")
    totals = end - start
    full_window = times + window <= OBSERVATION_END
    cumulative = np.cumsum(totals)
    prefix = np.zeros((n + 1, n_areas), dtype=np.int64)
    prefix[np.arange(n) + 1, area] = 1
    prefix = np.cumsum(prefix, axis=0)
    same_area_total = prefix[end, area] - prefix[start, area]
    poolable_all = np.zeros(n, dtype=np.int64)
    poolable_same = np.zeros(n, dtype=np.int64)
    flow_counts = np.zeros(n_areas * n_areas, dtype=np.int64)
    population = int(cumulative[-1]) if n else 0
    for low in range(0, population, chunk_size):
        ranks = np.arange(low, min(low + chunk_size, population))
        i = np.searchsorted(cumulative, ranks, side="right")
        previous = np.where(i > 0, cumulative[np.maximum(i - 1, 0)], 0)
        j = start[i] + ranks - previous
        positive = pair_rewards(origins, destinations, i, j) > 1e-12
        first, second = i[positive], j[positive]
        same = area[first] == area[second]
        poolable_all += np.bincount(first, minlength=n)
        poolable_same += np.bincount(first[same], minlength=n)
        eligible = full_window[first]
        flow_counts += np.bincount(
            area[first[eligible]] * n_areas + area[second[eligible]],
            minlength=n_areas * n_areas,
        )
    result = ordered[["day", "snapshot_row", "da_id", "time_seconds"]].copy()
    result["full_window"] = full_window
    result["future_arrivals_all_areas"] = totals
    result["future_arrivals_same_area"] = same_area_total
    result["poolable_all_areas"] = poolable_all
    result["poolable_same_area"] = poolable_same
    result["poolable_other_areas"] = poolable_all - poolable_same
    assert np.all(poolable_same <= same_area_total)
    assert np.all(poolable_all <= totals)
    assert np.all(poolable_same <= poolable_all)
    assert int(flow_counts.sum()) == int(poolable_all[full_window].sum())
    return result, flow_counts.reshape(n_areas, n_areas), population


def add_historical_density(jobs, raw_counts, window):
    totals = raw_counts.sum(axis=1)
    for day in raw_counts.columns:
        take = jobs.day.eq(day)
        historical = totals - raw_counts[day]
        jobs.loc[take, "historical_area_arrivals"] = jobs.loc[take, "da_id"].map(historical)
    jobs["historical_area_arrivals"] = jobs.historical_area_arrivals.astype(int)
    jobs["historical_arrival_rate_per_second"] = jobs.historical_area_arrivals / (7 * EXPOSURE_SECONDS)
    jobs["implied_area_density"] = jobs.historical_arrival_rate_per_second * window
    return jobs


def correlation(x, y):
    x, y = pd.Series(np.asarray(x)), pd.Series(np.asarray(y))
    return {
        "pearson": float(x.corr(y)),
        "spearman": float(x.rank().corr(y.rank())),
    }


def summarize(jobs, flows, areas, output):
    primary = jobs.loc[jobs.full_window].copy()
    area_day = primary.groupby(["da_id", "day"]).agg(
        jobs=("snapshot_row", "size"),
        implied_density=("implied_area_density", "first"),
        mean_poolable_all=("poolable_all_areas", "mean"),
        mean_poolable_same=("poolable_same_area", "mean"),
        mean_poolable_other=("poolable_other_areas", "mean"),
        mean_future_same_area=("future_arrivals_same_area", "mean"),
        fraction_no_poolable=("poolable_all_areas", lambda v: v.eq(0).mean()),
    ).reset_index()
    area_summary = primary.groupby("da_id").agg(
        jobs=("snapshot_row", "size"),
        mean_implied_density=("implied_area_density", "mean"),
        mean_poolable_all=("poolable_all_areas", "mean"),
        mean_poolable_same=("poolable_same_area", "mean"),
        mean_poolable_other=("poolable_other_areas", "mean"),
        mean_future_same_area=("future_arrivals_same_area", "mean"),
        fraction_no_poolable=("poolable_all_areas", lambda v: v.eq(0).mean()),
    ).reset_index()
    area_summary["share_poolable_other_areas"] = (
        area_summary.mean_poolable_other / area_summary.mean_poolable_all
    )
    area_summary["density_quartile"] = pd.qcut(
        area_summary.mean_implied_density, 4, labels=["Lowest", "Lower-middle", "Upper-middle", "Highest"]
    )
    primary["density_quartile"] = primary.da_id.map(area_summary.set_index("da_id").density_quartile)
    quartiles = primary.groupby("density_quartile", observed=True).agg(
        areas=("da_id", "nunique"), jobs=("snapshot_row", "size"),
        mean_implied_density=("implied_area_density", "mean"),
        mean_poolable_all=("poolable_all_areas", "mean"),
        mean_poolable_same=("poolable_same_area", "mean"),
        mean_poolable_other=("poolable_other_areas", "mean"),
    ).reset_index()
    daily_correlations = []
    for day, part in area_day.groupby("day"):
        daily_correlations.append({"day": int(day), **correlation(part.implied_density, part.mean_poolable_all)})
    metrics = {
        "snapshot_jobs": len(jobs), "complete_window_jobs": len(primary),
        "excluded_truncated_windows": int((~jobs.full_window).sum()),
        "mean_poolable_future_all_areas": float(primary.poolable_all_areas.mean()),
        "mean_poolable_future_same_area": float(primary.poolable_same_area.mean()),
        "mean_poolable_future_other_areas": float(primary.poolable_other_areas.mean()),
        "fraction_jobs_no_poolable_future": float(primary.poolable_all_areas.eq(0).mean()),
        "area_mean_correlations": correlation(area_summary.mean_implied_density, area_summary.mean_poolable_all),
        "area_mean_same_area_correlations": correlation(area_summary.mean_implied_density, area_summary.mean_poolable_same),
        "area_mean_observed_same_area_arrival_correlations": correlation(area_summary.mean_implied_density, area_summary.mean_future_same_area),
        "job_level_correlations": correlation(primary.implied_area_density, primary.poolable_all_areas),
        "daily_area_correlations": daily_correlations,
    }
    area_day.to_csv(output / "area_day_summary.csv", index=False)
    area_summary.sort_values("mean_implied_density").to_csv(output / "area_summary.csv", index=False)
    quartiles.to_csv(output / "density_quartiles.csv", index=False)
    # Matrix entry: average number of future poolable jobs in column area per
    # complete-window focal job in row area. Each row sums to mean_poolable_all.
    counts = pd.DataFrame(flows, index=areas, columns=areas)
    denominator = primary.groupby("da_id").size().reindex(areas)
    flow_means = counts.div(denominator, axis=0)
    flow_means.index.name = "focal_area"
    assert np.allclose(flow_means.sum(axis=1), area_summary.set_index("da_id").mean_poolable_all.reindex(areas))
    flow_means.to_csv(output / "mean_poolable_by_partner_area.csv")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def plot(output, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    areas = pd.read_csv(output / "area_summary.csv")
    day = pd.read_csv(output / "area_day_summary.csv")
    metrics = json.loads((output / "metrics.json").read_text())
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), gridspec_kw={"width_ratios": [1, 1.4]}, constrained_layout=True)
    axes[0].scatter(day.implied_density, day.mean_poolable_all, s=14, alpha=.25, color="#8c9baa", label="Area-day means")
    axes[0].scatter(areas.mean_implied_density, areas.mean_poolable_all, s=48, color="#245da8", label="Area means")
    for row in areas.itertuples():
        if row.da_id in {0, 5, 6, 9, 18, 22}:
            axes[0].annotate(str(row.da_id), (row.mean_implied_density, row.mean_poolable_all), xytext=(4, 5), textcoords="offset points", fontsize=9)
    spearman = metrics["area_mean_correlations"]["spearman"]
    axes[0].set(xscale="log", xlabel=f"Historical own-area arrival rate × {window:g} s", ylabel=f"Mean poolable arrivals from any area in {window:g} s", title=f"Denser area, more opportunities?  Spearman = {spearman:.2f}")
    axes[0].legend(frameon=False, fontsize=9)
    x = np.arange(len(areas))
    axes[1].bar(x, areas.mean_poolable_same, label="Same area", color="#245da8")
    axes[1].bar(x, areas.mean_poolable_other, bottom=areas.mean_poolable_same, label="Other areas", color="#e99b2d")
    axes[1].set_xticks(x, areas.da_id.astype(str))
    axes[1].set(xlabel="Area ID, ordered by increasing historical density", ylabel=f"Mean poolable arrivals per job in {window:g} s", title="Where the pooling opportunities come from")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Meituan area density and future pooling opportunities", fontsize=15)
    fig.supxlabel("Exact positive-reward counts; full waiting windows only. Historical density excludes the focal day. No prior dispatches are assumed.", fontsize=9)
    fig.savefig(output / "area_density_poolability.png", dpi=180)
    fig.savefig(output / "area_density_poolability.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raw", type=Path, default=Path("../data/all_waybill_info_meituan_0322.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/area_density_diagnostic"))
    parser.add_argument("--window", type=float, default=60)
    parser.add_argument("--plot-python")
    parser.add_argument("--plot-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.window < EXPOSURE_SECONDS:
        parser.error("window must lie between zero and the observation duration")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(output, args.window)
        return
    frame, audit, raw_rows, accepted_rows = load_labeled_snapshots(args.data_dir, args.raw)
    frame["snapshot_row"] = frame.groupby("day").cumcount()
    raw_counts = pd.crosstab(frame.da_id, frame.day).reindex(columns=range(8), fill_value=0)
    area_ids = sorted(frame.da_id.unique().tolist())
    results, flows, population = [], np.zeros((len(area_ids), len(area_ids)), dtype=np.int64), 0
    for day, part in frame.groupby("day"):
        counts, day_flows, pair_count = count_future_opportunities(part, args.window, area_ids)
        results.append(counts)
        flows += day_flows
        population += pair_count
        print(f"Day {day}: {len(part):,} jobs, {pair_count:,} future candidate pairs checked", flush=True)
    jobs = add_historical_density(pd.concat(results, ignore_index=True), raw_counts, args.window)
    jobs.to_csv(output / "job_opportunity_counts.csv.gz", index=False)
    raw_counts.to_csv(output / "area_daily_arrivals.csv")
    audit.to_csv(output / "join_audit.csv", index=False)
    metrics = summarize(jobs, flows, area_ids, output)
    provenance = {
        "raw_source": str(args.raw.resolve()), "snapshot_directory": str(args.data_dir.resolve()),
        "window_seconds": args.window, "candidate_pairs_checked": population,
        "raw_rows": raw_rows, "accepted_raw_rows": accepted_rows,
        "future_interval": "(arrival_time, arrival_time + window]; equal recorded timestamps excluded",
        "poolable": "current Euclidean all-pickups-before-deliveries reward > 1e-12",
        "historical_density": "other seven days' area arrival count / (7 * 10800 seconds) * window",
        "primary_sample": "focal arrivals with full window observed before 13:30",
        "counting": "all future jobs, regardless of area; no dispatch policy removes candidates",
        "correlations": "descriptive associations, with equal-area and job-level results separately",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if args.plot_python:
        subprocess.run([args.plot_python, str(Path(__file__).resolve()), "--plot-only", "--window", str(args.window), "--output-dir", str(output)], check=True)
    else:
        plot(output, args.window)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
