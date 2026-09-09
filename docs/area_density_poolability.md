# Area density and future pooling opportunities

The 60-second sanity check supports a positive association: jobs assigned to denser areas tend to encounter more poolable future jobs from the whole city. Area density is informative but does not determine the number of opportunities by itself.

## Definition and sample

For a focal job i assigned to area h on day t, compare:

```
rho[h, -t] = historical arrivals in area h / (7 * 10800 seconds) * 60 seconds
C[i]      = number of jobs j from ANY area with
            arrival[i] < arrival[j] <= arrival[i] + 60 seconds
            and pooling_reward(i, j) > 0
```

The historical sample is the other seven days. Poolability uses the current simulator's Euclidean distance savings with both pickups before either delivery, applying a numerical threshold of 1e-12. The computation checks every candidate pair exactly, rather than sampling. No dispatch policy removes jobs; C measures future opportunities over the allowed window, not realized partners or availability after earlier dispatches. Jobs already present at arrival and jobs with identical recorded timestamps are excluded from this prospective count.

All 190,875 jobs retain their recovered `da_id`. The main comparison uses 190,419 focal jobs with a fully observed 60-second window; 456 near the 13:30 end of the snapshot are excluded from summary statistics. Arrival rates use the complete three-hour snapshots, including those 456 jobs. The computation evaluates 29,754,395 strictly future candidate pairs across the eight days, including the observed portions of truncated focal windows. Counts for all focal jobs are retained with a full-window flag.

## Results

| Comparison | Pearson correlation | Spearman rank correlation |
| --- | ---: | ---: |
| Area density vs mean poolable arrivals from any area, 23 area means | 0.614 | 0.597 |
| Area density vs mean poolable arrivals from the same area, 23 area means | 0.861 | 0.945 |
| Area density vs mean total arrivals from the same area, 23 area means | 0.998 | 0.995 |
| Area density vs individual job's poolable arrivals from any area | 0.296 | 0.311 |

Each area has equal weight in the area-level correlations; the job-level calculation gives each job equal weight. Area summaries use job-weighted means across the eight days, with each job's own leave-one-day-out density. Across separate days, the area-level rank correlation with all-area pooling opportunities is positive on all eight days, ranging from 0.544 to 0.606. These are descriptive associations, not causal estimates or independent-fold significance tests.

Dividing the 23 areas into density quartiles gives:

| Area-density group | Areas | Focal jobs | Mean implied density | Mean poolable arrivals, any area | Same area | Other areas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lowest | 6 | 9,964 | 1.34 | 1.93 | 0.29 | 1.64 |
| Lower-middle | 6 | 26,572 | 3.26 | 4.20 | 0.68 | 3.53 |
| Upper-middle | 5 | 36,578 | 5.22 | 3.07 | 1.22 | 1.85 |
| Highest | 6 | 117,305 | 16.42 | 5.77 | 3.10 | 2.67 |

Quartiles group areas by their mean historical density; means within each group weight jobs equally. The highest group encounters approximately three times as many poolable future arrivals as the lowest group. The middle groups are not monotone.

Across all complete-window jobs, the mean is 4.83 poolable future arrivals: 2.25 from the same area and 2.58 from other areas. About 12.9% of jobs see none. Counts need not equal nominal density: nominal density counts all own-area arrivals, while the poolable count selects compatible jobs from the whole city. In addition, historical rates are three-hour averages and observed opportunities are averaged over actual job arrivals during varying lunch demand.

## Useful exceptions

| Area ID | Mean implied density | Mean poolable arrivals, any area | Same area | Other areas |
| --- | ---: | ---: | ---: | ---: |
| 5 | 14.61 | 8.96 | 5.10 | 3.86 |
| 9 | 14.71 | 3.37 | 1.82 | 1.56 |
| 6 | 9.44 | 2.58 | 2.58 | 0.00 |
| 12 | 3.64 | 6.08 | 1.01 | 5.07 |

Areas 5 and 9 have almost identical arrival densities but very different opportunity counts. Area 6 has no cross-area positive-reward future opportunities in this sample, while area 12 draws most of its opportunities from other areas. Spatial configuration and origin–destination compatibility therefore matter alongside own-area arrival intensity. These statistics alone do not establish which gamma is optimal.

![Area density and pooling opportunities](</Users/mer2262/Documents/Columbia/Research/Dynamic Delivery Pooling/ddp_pkg_release/results/area_density_diagnostic/area_density_poolability.png>)

## Implication for the coefficient experiment

Use the chosen job-level `da_id` to assign one coefficient per area, retaining citywide matching. A cross-area edge naturally uses each job's own shadow, so cross-area pooling does not prevent this policy definition.

Retain two historical descriptors when interpreting fitted gammas:

1. Nominal density: the area's arrival rate times the pooling window.
2. Historical effective opportunity count: mean C among that area's historical jobs, counting compatible arrivals from every area.

For target day t, the second descriptor must average counts from the other seven days only. The current all-eight-day plots are descriptive diagnostics, not a fitted runtime predictor. Keep the same complete-window convention when estimating the historical descriptor.

The coefficient table can vary freely by area initially; a density-only or opportunity-based functional rule is a later modeling choice. Citywide matching couples performance across coefficients, which affects how to tune them, but does not require changing the area assignment or excluding cross-area matches.

## Reproduction and validation

Run from the repository root with NumPy, pandas, and matplotlib installed:

```bash
python scripts/analyze_area_poolable_density.py --window 60
```

An alternate plotting Python may be specified with `--plot-python /path/to/python`. Source joins reuse the validated accepted-record recovery in `scripts/analyze_meituan_zones.py`. The original CSVs are read without modification.

Outputs under `results/area_density_diagnostic/` include:

- `job_opportunity_counts.csv.gz`: one row per snapshot job, with day, zero-based snapshot row, area ID, historical density, total/same-area/other-area poolable counts, and observation completeness. Coordinates are not exported.
- `area_summary.csv` and `area_day_summary.csv`: area and area-day comparisons.
- `mean_poolable_by_partner_area.csv`: mean number of compatible future jobs in each partner area per complete-window focal job. Every row sums to that focal area's mean total opportunity count.
- `density_quartiles.csv`, `metrics.json`, source provenance, join audit, and PNG/PDF plots.

Independent checks compared exact counts against brute-force calls to `ddp.model.reward` on a small instance, including equal timestamps, endpoint boundaries, multiple areas, and chunk boundaries. Historical-density checks confirmed that changing the focal day's area counts leaves that day's density estimates unchanged. Full-data checks reconcile same-area/other-area counts, candidate bounds, and partner-area matrix totals. The figure was rendered and inspected.
