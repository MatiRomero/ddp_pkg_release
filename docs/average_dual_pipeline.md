# Average-dual pipeline overview

This note links the hindsight-dual (HD) sampling utilities, type mappers, and runtime consumers that transform HD samples into average-dual (AD) shadows.

## 1. Generating HD samples

`ddp.scripts.build_hd_dataset` draws Monte-Carlo job instances and solves the LP relaxation once per instance. Each job contributes a row to the output CSV with the columns shown below:

| column | description |
| --- | --- |
| `instance_id` | Sequential identifier `0..instances-1`. |
| `job_index` | Position of the job in the generated list. |
| `seed` | RNG seed used when sampling the instance. |
| `timestamp` | Job arrival time (seconds). |
| `origin_x`, `origin_y` | Origin coordinates. |
| `dest_x`, `dest_y` | Destination coordinates. |
| `d` | Deadline / slack parameter passed to the LP. |
| `potential` | Baseline potential (`0.5 * job.length`). |
| `hindsight_dual` | Dual value returned by the LP relaxation. |

The script stores the rows via `_write_rows`, creating directories as needed before writing the CSV.【F:src/ddp/scripts/build_hd_dataset.py†L7-L107】 These datasets provide the raw HD duals that later get averaged by type.

## 2. Bucketing jobs into types

Average-dual tooling separates type assignment from the data pipeline. Coordinate-based schemes, such as the uniform spatial grid, treat each job as its `(origin, destination)` quadruple and return a type key describing the snapped grid cells. The built-in mapper lives in `ddp.mappings.uniform_grid`:

* `UniformGridMapping.__call__(origin_x, origin_y, dest_x, dest_y)` quantises coordinates using a configurable `type_width` and emits `((o_ix, o_iy), (d_ix, d_iy))` indices.【F:src/ddp/mappings/uniform_grid.py†L36-L83】
* When bounds are supplied, the mapper precomputes the Cartesian product of reachable cells and exposes them through an `expected_types` set so coverage can be analysed.【F:src/ddp/mappings/uniform_grid.py†L41-L75】

Because `load_average_dual_mapper` expects a callable that accepts a `Job` instance and returns a type string (or `None` when the job should be ignored), coordinate-based helpers must be wrapped before use. A thin adapter typically extracts the job fields and feeds them into the grid mapping, then serialises the tuple to a string. This distinction is important: **coordinate mappers operate on raw floats, while the runtime loader requires `Job -> type` callables.** The CLI resolver simply imports the provided `module:function` path and verifies that the object is callable.【F:src/ddp/scripts/run.py†L54-L82】

The inspection CLI (`ddp.scripts.average_duals`) resolves the same `module:callable` specification, reports the attached `expected_types`, and optionally lists them to help validate coverage assumptions.【F:src/ddp/scripts/average_duals.py†L9-L77】

## 3. Building AD lookup tables

After assigning types, aggregate HD rows into an average-dual table. Two storage formats are supported by `load_average_duals`:

* **NPZ archives** must contain parallel `types` and `mean_dual` (or `duals`) arrays. Each entry is coerced to `str` and mapped to `float` values.【F:src/ddp/scripts/run.py†L31-L52】
* **Delimited/CSV files** must provide a header row with `type` and `mean_dual` (or `dual`) columns. Extra columns, such as the
  `std_dev` summary emitted by `build_average_duals`, are ignored. Missing or invalid values raise
  `ValueError`.【F:src/ddp/scripts/run.py†L54-L81】【F:src/ddp/scripts/build_average_duals.py†L88-L117】

The resulting dictionary maps type strings to mean duals and can be shared across experiments.

## 4. Runtime consumption and fallback policies

At simulation time (`ddp.scripts.run`), AD shadows are enabled by passing both the table (`--ad_duals`) and the mapper (`--ad_mapping`). Jobs are mapped to type keys and filled with the corresponding averages. Missing entries trigger one of three policies controlled by `--ad_missing`:

* `hd` (default): fall back to the job’s original HD dual from the LP relaxation.
* `zero`: substitute `0.0`.
* `error`: abort with `AverageDualError` listing the unresolved types.

The helper `_resolve_average_duals` applies the mapping, records missing keys, and injects the requested fallback, returning the AD shadow vector alongside a coverage report.【F:src/ddp/scripts/run.py†L84-L150】 The dispatcher then proceeds exactly as with other shadow families: shadows may be rescaled (`gamma`, `tau`), re-used for the `+` variants, and evaluated against the standard score/weight functions.【F:src/ddp/scripts/run.py†L152-L303】

## Putting it all together

1. **Sample HD duals** with `python -m ddp.scripts.build_hd_dataset --out_csv hd_samples.csv ...` to produce the per-job CSV described above.
2. **Define or adapt a type mapper** that converts `Job` objects into type strings, possibly by wrapping a coordinate-based helper like the uniform grid.
3. **Aggregate by type** to compute mean duals, saving the lookup as an NPZ (`types`, `mean_dual`) or CSV (`type`, `mean_dual`, `std_dev`).
4. **Run experiments** with `python -m ddp.scripts.run --shadows ad --ad_duals ... --ad_mapping ... --ad_missing {hd,zero,error}` so AD shadows use the prepared table with the chosen fallback policy.

This pipeline keeps HD sampling, type assignment, and runtime evaluation decoupled, making it straightforward to iterate on new mappers or dataset slices without modifying the simulation logic.

## Inspecting coverage and variability

Use `python -m ddp.scripts.inspect_average_duals` to summarise an HD dataset before committing to a lookup table. The CLI shares the same mapping resolver as the rest of the pipeline and defaults to analysing the bundled `data/hd_dataset_n100_d10.csv` with the uniform grid mapper:

```bash
python -m ddp.scripts.inspect_average_duals \
  --mapping ddp.mappings.uniform_grid:mapping \
  --heatmap reports/uniform_grid_origin_coverage.png
```

The textual report prints one row per type with the sample count, mean dual, and standard deviation, mirroring the CSV produced by `build_average_duals`. When `--heatmap` is supplied the tool aggregates counts by origin grid cell and saves a simple matplotlib heatmap so you can visualise coverage gaps before training dispatch models.【F:src/ddp/scripts/inspect_average_duals.py†L7-L123】
