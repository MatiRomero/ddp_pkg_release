#!/usr/bin/env python3
from __future__ import annotations
import csv, time, argparse, math, os
from collections import defaultdict
import numpy as np
from typing import Dict, List, Any

from ddp.model import Job, generate_jobs
from ddp.scripts.run import run_instance  # ← replaced import

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics over many runs (SHADOW × DISPATCH × seeds)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--d", type=float, default=5.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0, help="starting seed (we use seed0..seed0+trials-1)")
    ap.add_argument("--shadows", default="naive,pb,hd")
    ap.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch")
    ap.add_argument("--outdir", default="results", help="Directory to save CSV (created if missing)")
    ap.add_argument("--save_csv", default="results_agg.csv", help="Filename (written inside --outdir)")
    ap.add_argument("--with_opt", action="store_true")
    ap.add_argument("--opt_method", default="auto", choices=["auto","networkx","ilp"])
    ap.add_argument(
        "--fix_origin_zero",
        action="store_true",
        help="Set every generated job's origin to the depot at (0, 0).",
    )
    ap.add_argument(
        "--flatten_axis",
        choices=["x", "y"],
        help=(
            "Project all jobs onto a single axis by zeroing the chosen coordinate "
            "for both origins and destinations."
        ),
    )

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, args.save_csv)

    shadows = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    dispatches = [d.strip().lower() for d in args.dispatch.split(",") if d.strip()]

    # Collect raw metrics per (shadow, dispatch)
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)

    t0 = time.perf_counter()
    pbar = tqdm(total=args.trials, desc="Trials", unit="trial") if (True and args.trials > 1 and tqdm is not None) else None
    if True and args.trials > 1 and tqdm is None:
        print("(Tip) Install tqdm for a progress bar: pip install tqdm")
    for t in range(args.trials):
        seed = args.seed0 + t
        rng = np.random.default_rng(seed)
        if args.n <= 1:
            raise ValueError("run_many requires n > 1 to generate jobs")
        jobs = generate_jobs(args.n, rng)
        if args.fix_origin_zero:
            jobs = [
                Job(origin=(0.0, 0.0), dest=job.dest, timestamp=job.timestamp)
                for job in jobs
            ]
        if args.flatten_axis is not None:
            axis = 0 if args.flatten_axis == "x" else 1

            def _flatten(point: tuple[float, float]) -> tuple[float, float]:
                coords = [point[0], point[1]]
                coords[axis] = 0.0
                return coords[0], coords[1]

            jobs = [
                Job(origin=_flatten(job.origin), dest=_flatten(job.dest), timestamp=job.timestamp)
                for job in jobs
            ]
        res = run_instance(jobs=jobs, d=args.d, shadows=shadows, dispatches=dispatches, seed=seed,
                           with_opt=args.with_opt, opt_method=args.opt_method, save_csv="", print_table=False,
                           return_details=False, print_matches=False)
        for rec in res["rows"]:
            buckets[(rec["shadow"], rec["dispatch"])].append(rec)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Determine the set of numeric metric keys (beyond id/meta fields)
    meta_keys = {"n","d","seed","shadow","dispatch"}
    all_keys = set()
    for rows in buckets.values():
        for rec in rows:
            for k, v in rec.items():
                if k in meta_keys:
                    continue
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    all_keys.add(k)
    all_keys = sorted(all_keys)  # stable order

    # Aggregate mean/std for each numeric key
    out_rows = []
    for (sh, disp), rows in buckets.items():
        row = {"shadow": sh, "dispatch": disp, "n": args.n, "d": args.d, "trials": args.trials, "seed0": args.seed0}
        for k in all_keys:
            xs = [float(r.get(k, float("nan"))) for r in rows]
            xs = [x for x in xs if not math.isnan(x)]
            if len(xs) == 0:
                mean = float("nan"); std = float("nan")
            elif len(xs) == 1:
                mean = xs[0]; std = 0.0
            else:
                mean = float(np.mean(xs))
                std = float(np.std(xs, ddof=1))
            row[f"mean_{k}"] = mean
            row[f"std_{k}"] = std
        out_rows.append(row)

    # Write CSV with dynamic columns (all discovered metrics)
    fieldnames = ["shadow","dispatch","n","d","trials","seed0"] + \
                 [f"mean_{k}" for k in all_keys] + [f"std_{k}" for k in all_keys]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    dur = time.perf_counter() - t0
    print(f"Wrote {os.path.abspath(save_path)} with {len(out_rows)} rows in {dur:.2f}s")
    # Optional: print which metrics were aggregated
    print("Aggregated metrics:", ", ".join(all_keys))

if __name__ == "__main__":
    main()