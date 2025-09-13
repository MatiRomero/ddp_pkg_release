import argparse, csv, time
import numpy as np
from ddp.scripts.run import run_instance


def _mean(x):
    return float(np.mean(x)) if len(x) else float('nan')

def _std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Generate random instance(s) then run SHADOW×DISPATCH. Supports multi-trial aggregation."
    )
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--d", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0, help="start seed; trials use seed..seed+trials-1")
    ap.add_argument("--trials", type=int, default=1, help="number of independent random instances to run and aggregate")
    ap.add_argument("--shadows", default="naive,pb,hd")
    ap.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch")
    ap.add_argument("--with_opt", action="store_true")
    ap.add_argument("--opt_method", default="auto", choices=["auto","networkx","ilp"])
    ap.add_argument("--save_csv", default="", help="if trials==1: per-run CSV; if trials>1: aggregated CSV")
    args = ap.parse_args()

    shadows = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    dispatches = [d.strip().lower() for d in args.dispatch.split(",") if d.strip()]

    if args.trials <= 1:
        # Single random instance (original behavior)
        rng = np.random.default_rng(args.seed)
        theta = rng.random(args.n)
        timestamps = np.arange(args.n, dtype=float)
        run_instance(
            theta=theta, timestamps=timestamps, d=args.d,
            shadows=shadows, dispatches=dispatches,
            seed=args.seed, with_opt=args.with_opt, opt_method=args.opt_method,
            save_csv=args.save_csv, print_table=True,
        )
        return

    # Multi-trial aggregation
    print(f"Aggregating {args.trials} trials: n={args.n}, d={args.d}, seeds {args.seed}..{args.seed+args.trials-1}")
    t0 = time.perf_counter()

    agg = {}  # (shadow, dispatch) -> dict of lists
    for t in range(args.trials):
        seed_t = args.seed + t
        rng = np.random.default_rng(seed_t)
        theta = rng.random(args.n)
        timestamps = np.arange(args.n, dtype=float)
        res = run_instance(
            theta=theta, timestamps=timestamps, d=args.d,
            shadows=shadows, dispatches=dispatches,
            seed=seed_t, with_opt=args.with_opt, opt_method=args.opt_method,
            save_csv="", print_table=False,
        )
        # accumulate rows
        for row in res["rows"]:
            key = (row["shadow"], row["dispatch"]) 
            bucket = agg.setdefault(key, {
                "savings": [], "pooled": [], "pairs": [], "solos": [],
                "ratio_lp": [], "ratio_opt": []
            })
            bucket["savings"].append(row["savings"])
            bucket["pooled"].append(row["pooled_pct"])
            bucket["pairs"].append(row["pairs"])
            bucket["solos"].append(row["solos"])
            if row.get("ratio_lp") is not None:
                bucket["ratio_lp"].append(row["ratio_lp"])
            if args.with_opt and (row.get("ratio_opt") is not None):
                bucket["ratio_opt"].append(row["ratio_opt"])

    # print summary table
    print("SHADOW     DISPATCH     MEAN POOLED%   MEAN SAVINGS   ±STD      MEAN R/LP   MEAN R/OPT")
    rows_out = []
    for (sh, disp), data in sorted(agg.items()):
        mean_sav, std_sav = _mean(data["savings"]), _std(data["savings"])
        mean_pool = _mean(data["pooled"])
        mean_pairs, mean_solos = _mean(data["pairs"]), _mean(data["solos"])
        mean_rlp = _mean(data["ratio_lp"]) if data["ratio_lp"] else float('nan')
        mean_ropt = _mean(data["ratio_opt"]) if (args.with_opt and data["ratio_opt"]) else float('nan')
        print(f"{sh.upper():<10} {disp:<12} {mean_pool:11.1f}%   {mean_sav:12.3f}  ±{std_sav:6.3f}    {mean_rlp:7.3f}x    {mean_ropt:7.3f}x")
        row_out = {
            "shadow": sh, "dispatch": disp, "n": args.n, "d": args.d,
            "trials": args.trials, "seed0": args.seed,
            "mean_savings": mean_sav, "std_savings": std_sav,
            "mean_pooled_pct": mean_pool, "mean_pairs": mean_pairs, "mean_solos": mean_solos,
            "mean_ratio_lp": mean_rlp,
        }
        if args.with_opt:
            row_out["mean_ratio_opt"] = mean_ropt
        rows_out.append(row_out)

    # write aggregated CSV if requested
    if args.save_csv:
        fieldnames = list(rows_out[0].keys()) if rows_out else ["shadow","dispatch","n","d","trials","seed0"]
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows_out)
        print(f"Wrote aggregated CSV to {args.save_csv}")

    print(f"Done in {time.perf_counter()-t0:.2f}s")


if __name__ == "__main__":
    main()