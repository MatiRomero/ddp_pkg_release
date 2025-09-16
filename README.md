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
python -m ddp.scripts.plot_many --csv results/results_agg.csv --outdir figs --with_std
```

This creates one heatmap grid per metric (saved as PNGs in `figs/`), annotated with mean ± std values. A `summary_sorted.csv` file is also written for quick inspection.

- Reward (toy): `min(theta[i], theta[j])`
- Time model: `timestamps` + `time_window` (scalar or per-job)
- Batch/rbatch use critical-aware weights so for critical `i`: weight(i,j) = reward(i,j) - s_j
