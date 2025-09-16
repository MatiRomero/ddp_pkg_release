# Dynamic Delivery Pooling — SHADOW × DISPATCH (Package)

This project provides a research package for experimenting with online delivery pooling algorithms using SHADOW × DISPATCH heuristics. It includes tools to simulate, evaluate, and aggregate performance across multiple algorithmic strategies.

The core data structure is a `Job` dataclass (`ddp.model.Job`) that records each request's origin, destination, timestamp, and implied travel length.  Simulations and analytics operate directly on these job objects, making it easy to experiment with different spatial layouts and pooling policies.

Install in editable mode, then use the CLI entry points.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Optional plotting dependencies for ``ddp-mwe01 --plot`` and other figures
pip install -e .[plot]
```

## Run
```bash
ddp-mwe01
ddp-many  --trials 20 --n 100 --d 2 --save_csv results_agg.csv
```

## Aggregate and Plot Results

To run multiple trials and aggregate results across all shadow × dispatch combinations:

```bash
python -m ddp.scripts.run_many --trials 20 --n 100 --d 2 --outdir results --save_csv results_agg.csv
# Optional geometry shortcuts:
#   --fix_origin_zero        → pin every origin at the depot (0, 0)
#   --flatten_axis {x,y}     → collapse jobs onto a single axis (1-D experiments)
```

This will write an aggregated CSV (e.g., `results/results_agg.csv`) with mean and std columns for every metric.

To generate plots from this CSV:

```bash
python -m ddp.scripts.plot_results --csv_agg results/results_agg.csv --mode grid --outdir figs --metric all
```

This creates one heatmap grid per metric (saved as PNGs in `figs/`), annotated with mean ± std values when available.

- Reward (toy): `min(theta[i], theta[j])`
- Time model: `timestamps` + `time_window` (scalar or per-job)
- Batch/rbatch use critical-aware weights so for critical `i`: weight(i,j) = reward(i,j) - s_j

## Parameter Sweep Examples

Use the unified sweep runner to stream per-trial CSV rows while exploring different geometry and time window settings. Each command below writes its results to `sweeps/` (created automatically) and computes the optimal baseline alongside the heuristics.

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
