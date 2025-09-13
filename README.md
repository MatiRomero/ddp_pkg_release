# Dynamic Delivery Pooling — SHADOW × DISPATCH (Package)

Install in editable mode, then use the CLI entry points.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Run
```bash
ddp-mwe01
ddp-many  --trials 20 --n 500 --d 5 --save_csv results_agg.csv
ddp-demo  --shadows naive,pb,hd --dispatch greedy,greedy+,batch,rbatch --n 500 --d 5 --seed 0 --with_opt
```

- Reward (toy): `min(theta[i], theta[j])`
- Time model: `timestamps` + `time_window` (scalar or per-job)
- Batch/rbatch use critical-aware weights so for critical `i`: weight(i,j) = reward(i,j) - s_j
