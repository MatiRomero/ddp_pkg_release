# Preliminary Meituan zoning diagnostics

Analysis date: 2026-09-08. Population: the 190,875 jobs in the eight existing city snapshots, 10:30–13:30 on October 17–24, 2022. These are descriptive diagnostics; no coefficient was fitted and no dispatch policy was run.

Follow-up decision: use the supplied area ID for the first coefficient experiment, retaining citywide matching. The subsequent [per-job density sanity check](area_density_poolability.md) tests whether own-area arrival density predicts poolable arrivals from every area. Geographic endpoint assignments are not required for the selected area-ID policy.

## Findings

| Grouping | Assigned/origin zones | Destination zones | Median jobs per zone per day | Orders with destination outside origin zone |
| --- | ---: | ---: | ---: | ---: |
| Area ID (`da_id`) | 23 | Unavailable | 738.8 | Unavailable |
| H3 resolution 7 | 38 | 73 | 202.5 | 63.0% |
| H3 resolution 8 | 155 | 329 | 50.9 | 89.5% |
| H3 resolution 9 | 541 | 1,477 | 15.9 | 97.6% |

For each zone, daily volume is its total across the eight snapshots divided by eight; the table reports the median of those zone means. All 23 area IDs appear on all eight days. Area-level daily means range from 29.6 to 4,677.6 jobs. At resolution 8, 131 of 155 origin cells appear on all eight days. Resolution 8 has 6,236 observed OD pairs across the sample, versus 574 at resolution 7 and 41,979 at resolution 9.

The H3 OD crossing results are stable across days: 62.7–63.3% at resolution 7, 89.2–89.8% at resolution 8, and 97.5–97.9% at resolution 9. Area IDs provide considerably more observations per coefficient, while resolution 7 offers an intermediate geographic grouping worth retaining in the comparison.

![Zoning diagnostics](/Users/mer2262/Documents/Columbia/Research/Dynamic%20Delivery%20Pooling/ddp_pkg_release/results/zone_diagnostics/zone_diagnostics.png)

## Cross-zone pooling opportunities

An order crossing from its origin zone to its destination zone does not imply that it is pooled with an order assigned to a different zone. To examine the latter question, sample candidate pairs independently of the dispatch policy.

There are 30,023,438 unordered pairs of jobs from the same day with arrival times within 60 seconds. A uniform sample of 100,000 pairs with replacement, using seed 20260908, contains 3,153 pairs with strictly positive reward under the current simulator's Euclidean distance model. Each pair's route visits both origins before either destination, exactly as in the simulator. The calculations do not use road distances or additional operational restrictions.

| Job grouping | Different zones among positive-reward sampled pairs | Share of summed positive reward on cross-zone pairs |
| --- | ---: | ---: |
| Area ID | 53.5% | 46.9% |
| H3 resolution 7, origin | 45.1% | 40.3% |
| H3 resolution 8, origin | 75.2% | 71.4% |
| H3 resolution 9, origin | 89.3% | 86.8% |

These are approximate candidate-pair statistics, not selected matches, independent job outcomes, or estimates of the percentage of policy savings lost by partitioning. Many candidate edges compete for the same jobs, and the sample does not account for prior dispatches. Nevertheless, under the current model, neither area IDs nor resolution-8 origin cells form independent markets. Area IDs retain more candidate opportunities within groups than resolution 8; resolution 7 retains more than either in this sample.

## What area ID currently tells us

The raw data supplies one `da_id` per dispatch record. There are no separate sender-area and recipient-area columns, and no boundary file was found in the inspected project data. An area-to-area OD matrix is therefore unavailable from these fields alone. Assigning the record's one label to both endpoints would manufacture a zero crossing rate.

Area IDs also overlap geographically at the H3 scale. If each resolution-8 origin cell is labeled by its most frequent area ID, 70.0% of jobs agree with that label. Applying the same diagnostic separately to destination cells gives 46.8% agreement. These are in-sample descriptive agreements, not validated geographic classifiers. Area ID should not be assumed to be a simple coarsening of the H3 partition.

This overlap also occurs at identical recorded coordinates: 2,153 of 4,143 sender locations appear with multiple area IDs across the eight days. Those locations account for 75.4% of jobs. A modal-area lookup at the exact sender coordinate agrees with 86.3% of job labels. At exact recipient coordinates, the analogous agreement is 87.5%. These descriptive checks reinforce the need to establish what the area label represents before constructing geographic area boundaries from it.

If `da_id` is a service-area label known at order arrival, it could directly index a job's coefficient; origin versus destination assignment would then be a question specific to the geographic alternatives. If it is determined later by courier assignment, it needs an arrival-time definition before it can be used by an online policy. Its exact semantics and boundary availability remain to be confirmed.

## Data recovery and checks

The simulation CSVs dropped `da_id`. Their exported row index is not a reliable raw-data join key. The original notebook filters `is_courier_grabbed == True`, divides coordinates by 1,000,000, and converts Unix timestamps to local time by adding eight hours.

This analysis follows that accepted-record filter, then joins snapshots to raw records using all four coordinates (integer microdegrees), `dt`, `platform_order_time`, and `order_push_time`. Every snapshot row receives an unambiguous area label. Eight snapshot rows have multiple matching accepted raw records with the same area label; those keys are collapsed for the label lookup without adding or removing snapshot jobs. No area label is selected from a conflicting match.

Checks reconcile every zone-count table to 190,875 jobs, preserve all eight snapshots, verify observation-window bounds, and ensure sampled pairs are distinct same-day jobs within the declared time gap. The vectorized reward calculation was checked against explicit enumeration of all four possible pickup/dropoff routes for 20 sampled pairs. The figure was rendered and visually inspected.

Sources are the existing `data/meituan_city_lunchtime_plat10301330_day{0..7}.csv` snapshots and the parent project's `data/all_waybill_info_meituan_0322.csv`. Raw preprocessing is documented in cells 38, 50, and 100 of the parent project's `Dynamic Pooling Simulation/meituan_data_preliminary_analysis_II.ipynb`. Full local source paths and calculation definitions are recorded in the generated provenance file.

## Reproduction and next step

From the repository root, with NumPy, pandas, real H3, and matplotlib available:

```bash
python scripts/analyze_meituan_zones.py
```

If H3 or matplotlib is in another Python environment, provide `--h3-python /path/to/python` or `--plot-python /path/to/python`. The analysis writes aggregate CSVs, source metadata, and PNG/PDF figures under `results/zone_diagnostics/`. It does not overwrite source datasets or export individual enriched orders.

The first experiment now uses a job's supplied `da_id` directly, retaining cross-area matching. Geographic alternatives remain optional comparisons. Coefficient fitting and evaluation should use the intended citywide matching scope. The separate [synthetic GRID experiment](synthetic_gamma_grid.md), finalized for heterogeneous origins in 2D, is ready to run independently.
