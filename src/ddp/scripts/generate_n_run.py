#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
from ddp.scripts.generate_n_run import run_trials

"""
run_many.py — thin wrapper

Keeps backward-compatible flags, but delegates actual work to generate_n_run.run_trials().
This removes duplicated logic while preserving your CLI ergonomics.
"""

def main():
    ap = argparse.ArgumentParser(description="Aggregate metrics over many runs (SHADOW × DISPATCH × seeds)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--d", type=float, default=5.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0, help="starting seed (we use seed0..seed0+trials-1)")
    ap.add_argument("--shadows", default="naive,pb,hd")
    ap.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch")
    ap.add_argument("--with_opt", action="store_true")
    ap.add_argument("--opt_method", default="auto", choices=["auto","networkx","ilp"])
    ap.add_argument("--outdir", default="results", help="Directory to save CSV (created if missing)")
    ap.add_argument("--save_csv", default="results_agg.csv", help="Filename (written inside --outdir)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, args.save_csv) if args.save_csv else ""

    # Delegate to the single source of truth
    run_trials(
        n=args.n,
        d=args.d,
        seed=args.seed0,
        trials=args.trials,
        shadows=args.shadows,
        dispatch=args.dispatch,
        with_opt=args.with_opt,
        opt_method=args.opt_method,
        save_csv=save_path,
        print_table=True,
    )

if __name__ == "__main__":
    main()