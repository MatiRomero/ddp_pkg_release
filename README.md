# Dynamic Delivery Pooling — SHADOW × DISPATCH (Package)

This project provides a research package for experimenting with online delivery pooling algorithms using SHADOW × DISPATCH heuristics. It includes tools to simulate, evaluate, and aggregate performance across multiple algorithmic strategies.

The core data structure is a `Job` dataclass (`ddp.model.Job`) that records each request's origin, destination, timestamp, and implied travel length.  Simulations and analytics operate directly on these job objects, making it easy to experiment with different spatial layouts and pooling policies.

Install in editable mode, then use the CLI entry points.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Optional plotting dependencies for ``ddp-mwe01 --plot`` / ``ddp-mwe02 --plot`` and other figures
pip install -e .[plot]
```

## Run
```bash
ddp-mwe01
ddp-mwe02
ddp-many  --trials 20 --n 100 --d 2 --save_csv results_agg.csv
# Batch-convert the Meituan lunchtime CSV dumps into .npz archives and sweep d
python scripts/run_meituan_sweep.py
# Load jobs straight from a Meituan-style CSV (see notes below)
python -m ddp.scripts.run --jobs-csv data/meituan_sample.csv --d 3 \
  --export-npz meituan_sample.npz --save_csv run_from_csv.csv
# Inspect a stored instance (origins/dests/timestamps npz) and follow the available set
ddp-trace-available --jobs sample_instance.npz --d 3 --policy rbatch --shadow pb --plot
# Evaluate average-dual (AD) shadows using a pre-computed table and type mapper
python -m ddp.scripts.run \
  --jobs sample_instance.npz --d 3 --shadows ad --dispatch batch,rbatch \
  --ad_duals ad_means.npz --ad_mapping my_project.mappers:job_type --ad_missing hd
# Built-in uniform grid mapper (floors each origin/destination into 0.1-unit cells)
python -m ddp.scripts.average_duals --mapping ddp.mappings.uniform_grid:mapping --show-types
# Aggregate a hindsight-dual dataset into an average-dual CSV using the uniform grid mapper
python -m ddp.scripts.build_average_duals data/hd_samples.csv \
  ddp.mappings.uniform_grid:mapping data/ad_uniform_grid.csv
# Inspect coverage/variability for a dataset and (optionally) export an origin heatmap
python -m ddp.scripts.inspect_average_duals data/hd_dataset_n100_d10.csv \
  --mapping ddp.mappings.uniform_grid:mapping \
  --heatmap reports/uniform_grid_origin_coverage.png
```

Average-dual (``ad``) shadows map each job to a discrete type (via
``module:function`` provided to ``--ad_mapping``) and pull a mean dual value from
the lookup passed to ``--ad_duals``. Tables may be ``.npz`` archives containing
parallel ``types`` and ``mean_dual`` arrays or CSV files with ``type``, ``mean_dual``,
and ``std_dev`` columns. When a job's mapped type is absent, specify the fallback
behaviour with ``--ad_missing``: ``hd`` (default) replaces that job's shadow with
the HD dual from the LP relaxation, ``zero`` substitutes 0, and ``error`` aborts.
The same options are available for ``ddp-trace-available`` and
``python -m ddp.scripts.sweep_param`` so interactive tracing and sweep runs can
share the same assets. See [Average-dual pipeline overview](docs/average_dual_pipeline.md)
for a detailed walkthrough of the HD sampling, type mapping, and runtime integration pipeline.

### Loading job instances from CSV files

`python -m ddp.scripts.run` now accepts ``--jobs-csv`` to ingest the Meituan
sample CSV (or any file that matches its schema). By default the loader expects
ISO 8601 timestamps in the ``platform_order_time`` column; override
``--timestamp-column`` if your file stores these values under a different name
(for example switch to ``order_push_time`` when you want the order release
moment). Timestamps are normalised by subtracting the earliest valid record so
the first order always sits at ``t = 0`` s and every subsequent job is measured
in elapsed seconds. Because the simulator works in seconds, ensure any
deadlines ``d`` supplied via the CLI use the same unit. Rows with missing
coordinates or malformed timestamps are skipped silently so cleaning can happen
upstream.

Use ``--export-npz`` to persist the parsed jobs as an ``origins``/``dests``/
``timestamps`` archive alongside your aggregated CSV. The resulting ``.npz``
files remain compatible with all existing tracing and plotting tools (e.g.
``ddp-trace-available`` and ``ddp.scripts.plot_results``), so you can mix CSV-
backed and legacy workflows without changes.

Example end-to-end run on the bundled Meituan sample:

```bash
python -m ddp.scripts.run \
  --jobs-csv data/meituan_sample.csv \
  --timestamp-column platform_order_time \
  --d 3 \
  --export-npz data/meituan_sample.npz \
  --save_csv data/meituan_sample_summary.csv
```

To sanity-check a custom CSV, point the loader at the file and specify the key
flags explicitly so you can confirm the normalised timestamps match your
expectations:

```bash
python -m ddp.scripts.run \
  --jobs-csv my_orders.csv \
  --timestamp-column order_push_time \
  --export-npz my_orders.npz \
  --save_csv my_orders_summary.csv
```

The loader interprets timestamps with explicit offsets (e.g. ``+08:00``) and
converts trailing ``Z`` markers to ``+00:00`` automatically; ensure any local
timezones are encoded directly in the CSV if you need precise alignment with
external datasets.

## Aggregate and Plot Results

To run multiple trials and aggregate results across all shadow × dispatch combinations:

```bash
python -m ddp.scripts.run_many --trials 20 --n 100 --d 2 --outdir results --save_csv results_agg.csv
# Optional geometry shortcuts:
#   --fix_origin_zero        → pin every origin at the depot (0, 0)
#   --flatten_axis {x,y}     → collapse jobs onto a single axis (1-D experiments)
```

The default dispatch set covers `greedy`, `greedy+`, `batch`, `batch+`, `rbatch`, and
`rbatch+`. The baseline BATCH/RBATCH heuristics now default to scaling shadows by
γ = 0.5 (still overridable via `--gamma`/`--tau`). The `+` variants apply a
"late-arrival" adjustment: when pairing jobs they subtract only the later job's
shadow value, using separate `--plus_gamma`/`--plus_tau` controls (defaults
`--plus_gamma = 1`, `--plus_tau = 0`). The resulting weight is strictly
`reward(i, j) - s_late` and the simulation no longer re-adds the critical shadow.

This will write an aggregated CSV (e.g., `results/results_agg.csv`) with mean and std columns for every metric.

To generate plots from this CSV:

```bash
python -m ddp.scripts.plot_results --csv_agg results/results_agg.csv --mode grid --outdir figs --metric all
```

This creates one heatmap grid per metric (saved as PNGs in `figs/`), annotated with mean ± std values when available.

- Reward (toy): `min(theta[i], theta[j])`
- Time model: `timestamps` + `time_window` (scalar or per-job)
- Batch/rbatch subtract both job shadows when scoring candidate pairs.
- Batch+/rbatch+ use late-arrival weights so for jobs `(i, j)` the later arrival's
  shadow is subtracted: `weight(i, j) = reward(i, j) - s_late`.

## Parameter Sweep Examples

Use the unified sweep runner to stream per-trial CSV rows while exploring different geometry and time window settings. Each command below writes its results to `sweeps/` (created automatically) and computes the optimal baseline alongside the heuristics. When `--with_opt` is supplied the offline OPT baseline receives the same deadline parameter `d` (scalar or per-job array) and discards pairs whose availability windows do not overlap.

### 1. 1-D sweep with fixed origins and flattened x-axis

Run `n=100` jobs while sweeping `d` from 1 through 10 (inclusive). Every origin is pinned at the depot and coordinates are flattened onto the x-axis for a 1-D layout:

```bash
python -m ddp.scripts.sweep_param \
  --param d --values 1:10:1 \
  --n 100 \
  --with_opt \
  --fix_origin_zero \
  --flatten_axis x \
  --outdir sweeps \
  --save_csv n100_d1-10_opt_1d.csv
```

### 2. 2-D sweep with fixed origins

Sweep the same `d=1..10` range for `n=100` jobs without flattening destinations, keeping only the depot constraint:

```bash
python -m ddp.scripts.sweep_param \
  --param d --values 1:10:1 \
  --n 100 \
  --with_opt \
  --fix_origin_zero \
  --outdir sweeps \
  --save_csv n100_d1-10_opt_2d.csv
```

### 3. 2-D sweep over larger windows

For larger instances (`n=1000`) sweep `d` from 10 to 100 in steps of 10. This keeps the full 2-D geometry (no flattening) while computing OPT:

```bash
python -m ddp.scripts.sweep_param \
  --param d --values 10:100:10 \
  --n 1000 \
  --with_opt \
  --outdir sweeps \
  --save_csv n1000_d10-100_opt_2d.csv
```

Adjust `--trials`, `--shadows`, or `--dispatch` as needed for your experiments. CSV rows are flushed as each `(shadow, dispatch, trial)` completes, so you can stop the run early and still inspect partial results.

To visualise the sweep trends after aggregation:

```bash
python -m ddp.scripts.plot_results --csv_full sweeps/n100_d1-10_opt_1d.csv --mode sweep --metric mean_savings
```

Use `--include_policies` (comma-separated `shadow+dispatch` keys) to restrict which policy curves are shown.
