# src/ddp/scripts/sweep_param.py
import argparse, csv, time, os
import numpy as np
from ddp.scripts.run import run_instance
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def _mean(x):
    return float(np.mean(x)) if len(x) else float('nan')

def _std(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _parse_values(vals: str):
    """Parse "1,2,3" or a range like "1:5:1" (floats allowed)."""
    vals = vals.strip()
    if ":" in vals:
        a, b, *rest = vals.split(":")
        step = float(rest[0]) if rest else 1.0
        a, b = float(a), float(b)
        # inclusive range
        n = int(round((b - a) / step)) + 1
        return [a + i * step for i in range(n)]
    return [float(x) for x in vals.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Sweep a parameter (d or n) over values; aggregate trials per value."
    )
    ap.add_argument("--param", choices=["d", "n"], required=True,
                    help="which parameter to sweep")
    ap.add_argument("--values", required=True,
                    help='comma list "1,2,3" or range "1:5:1" (floats allowed)')
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="start seed; uses seed..seed+trials-1")
    ap.add_argument("--n", type=int, default=100, help="fixed n when sweeping d")
    ap.add_argument("--d", type=float, default=5.0, help="fixed d when sweeping n")
    ap.add_argument("--shadows", default="naive,pb,hd")
    ap.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch")
    ap.add_argument("--with_opt", action="store_true")
    ap.add_argument("--opt_method", default="auto", choices=["auto","networkx","ilp"])
    ap.add_argument("--save_csv", default="results/sweep_results.csv")
    args = ap.parse_args()

    # Normalize save path: if no directory provided, put it under results/
    save_path = args.save_csv
    if save_path:
        dirpart = os.path.dirname(save_path)
        if not dirpart:
            save_path = os.path.join("results", save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    shadows = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    dispatches = [d.strip().lower() for d in args.dispatch.split(",") if d.strip()]
    values = _parse_values(args.values)

    if args.param == "d":
        print(f"Sweeping d over {values} | trials={args.trials} | n={args.n} (fixed)")
    else:
        print(f"Sweeping n over {values} | trials={args.trials} | d={args.d} (fixed)")

    t0 = time.perf_counter()

    out_rows = []
    for v in values:
        agg = {}  # (shadow,dispatch) -> dict of lists
        # Resolve n,d once per parameter value
        if args.param == "d":
            n = args.n
            d = float(v)
        else:
            n = int(round(v))
            d = args.d
        # Single header per value
        print(f"\n== Sweeping {args.param} = {v} (n={n}, d={d}) | trials={args.trials} ==")
        # Progress bar per value
        if args.trials > 1 and tqdm is None:
            print("(Tip) Install tqdm for a progress bar: pip install tqdm")
        pbar = tqdm(total=args.trials, desc=f"{args.param}={v}", unit="trial") if (args.trials > 1 and tqdm is not None) else None
        for t in range(args.trials):
            seed_t = args.seed + t

            rng = np.random.default_rng(seed_t)
            theta = rng.random(n)
            timestamps = np.arange(n, dtype=float)

            res = run_instance(
                theta=theta, timestamps=timestamps, d=d,
                shadows=shadows, dispatches=dispatches,
                seed=seed_t, with_opt=args.with_opt, opt_method=args.opt_method,
                save_csv="", print_table=False,
            )
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # After trials, write one row per (shadow,dispatch) with means and stds for ALL metrics
        for (sh, disp), data in sorted(agg.items()):
            mean_sav, std_sav = _mean(data["savings"]), _std(data["savings"])
            mean_pool, std_pool = _mean(data["pooled"]), _std(data["pooled"])
            mean_rlp = _mean(data["ratio_lp"]) if data["ratio_lp"] else float('nan')
            std_rlp  = _std(data["ratio_lp"])  if len(data["ratio_lp"]) > 1 else 0.0
            if args.with_opt and data["ratio_opt"]:
                mean_ropt = _mean(data["ratio_opt"])
                std_ropt  = _std(data["ratio_opt"]) if len(data["ratio_opt"]) > 1 else 0.0
            else:
                mean_ropt = float('nan')
                std_ropt  = 0.0

            out_rows.append({
                "param": args.param,
                "param_value": float(v),
                "trials": args.trials,
                "seed0": args.seed,
                "n": (n if args.param == "n" else args.n),
                "d": (d if args.param == "d" else args.d),
                "shadow": sh,
                "dispatch": disp,
                # means
                "mean_savings": mean_sav,
                "mean_pooled_pct": mean_pool,
                "mean_ratio_lp": mean_rlp,
                **({"mean_ratio_opt": mean_ropt} if args.with_opt else {}),
                # stds
                "std_savings": std_sav,
                "std_pooled_pct": std_pool,
                "std_ratio_lp": std_rlp,
                **({"std_ratio_opt": std_ropt} if args.with_opt else {}),
            })

    # write CSV
    if out_rows:
        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"\nWrote {len(out_rows)} rows to {save_path}")
    print(f"Done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()