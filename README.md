# Dynamic Delivery Pooling — SHADOW × DISPATCH (Package)

This project provides a research package for experimenting with online delivery pooling algorithms using SHADOW × DISPATCH heuristics. It includes tools to simulate, evaluate, and aggregate performance across multiple algorithmic strategies.

Install in editable mode, then use the CLI entry points.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run
```bash
ddp-mwe01
ddp-many  --trials 20 --n 100 --d 2 --save_csv results_agg.csv
ddp-demo  --shadows naive,pb,hd --dispatch greedy,greedy+,batch,rbatch --n 100 --d 2 --seed 0 --with_opt
```

## Aggregate and Plot Results

To run multiple trials and aggregate results across all shadow × dispatch combinations:

```bash
python -m ddp.scripts.run_many --trials 20 --n 100 --d 2 --outdir results --save_csv results_agg.csv
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
