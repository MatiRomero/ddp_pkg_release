import time, csv, argparse, numpy as np

from ddp.engine.sim import simulate
from ddp.engine.opt import compute_opt, compute_lp_relaxation
from ddp.algorithms.potential import potential as potential_vec

def reward_fn(i, j, theta):
    # Toy 1D distance-savings
    return min(theta[i], theta[j])

def make_local_score(reward_fn, sp):
    # score for local search (asymmetric): r(i,j) - s_j
    def score(i, j, theta):
        return reward_fn(i, j, theta) - float(sp[j])
    return score

def make_weight_fn(reward_fn, sp):
    # matching edge weight (symmetric): r(i,j) - s_i - s_j
    def w(i, j, theta):
        return reward_fn(i, j, theta) - float(sp[i]) - float(sp[j])
    return w

def _safe_gap(upper, val): return max(upper - val, 0.0)

def _write_csv(rows, path):
    if not path: return
    fields = ["shadow","dispatch","n","d","seed","savings","pooled_pct",
              "ratio_lp","lp_gap","ratio_opt","opt_gap","pairs","solos","time_s","method"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description="SHADOW × DISPATCH with LP bound & timings.")
    p.add_argument("--shadows", default="naive,pb,hd", help="Comma list: naive,pb,hd")
    p.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch", help="Comma list: greedy,greedy+,batch,rbatch")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--d", type=float, default=5.0, help="time window (scalar)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with_opt", action="store_true", help="Also compute offline OPT on reward graph.")
    p.add_argument("--opt_method", default="auto", choices=["auto","networkx","ilp"])
    p.add_argument("--save_csv", default="")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    theta = rng.random(args.n)
    timestamps = np.arange(args.n, dtype=float)

    # LP (upper bound) + duals (for 'hd')
    t0 = time.perf_counter()
    lp = compute_lp_relaxation(theta, reward_fn)
    lp_time = time.perf_counter() - t0
    lp_total, duals = lp["total_upper"], np.array(lp["duals"], dtype=float)
    print(f"LP_RELAX  upper={lp_total:.3f}  frac_edges={len(lp['frac_edges']):>4}  method={lp['method']}  time={lp_time:.3f}s")

    # Optional OPT on reward graph
    opt_total = None; opt_pairs = None; opt_method = None; opt_time = None
    if args.with_opt:
        try:
            t0 = time.perf_counter()
            opt = compute_opt(theta, reward_fn, method=args.opt_method)
            opt_time = time.perf_counter() - t0
            opt_total, opt_pairs, opt_method = opt["total_reward"], opt["pairs"], opt["method"]
            print(f"OPT       total={opt_total:.3f}  pairs={len(opt_pairs):>4}  method={opt_method}  time={opt_time:.3f}s")
        except Exception as e:
            print(f"OPT       unavailable ({e})")

    # Parse SHADOW and DISPATCH lists
    shadows = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    dispatches = [d.strip().lower() for d in args.dispatch.split(",") if d.strip()]

    print(f"\nToy run: n={args.n}, time_window d={args.d}")
    print("-" * 142)
    print("SHADOW     DISPATCH     POOLED%   SAVINGS    R/LP   LP_GAP    R/OPT  OPT_GAP   #PAIRS  #SOLOS   TIME(s)")

    rows = []
    for sh in shadows:
        # shadow price vector
        if sh == "naive":
            sp = np.zeros(args.n, dtype=float)
        elif sh == "pb":
            sp = potential_vec(theta)
        elif sh == "hd":
            sp = duals
        else:
            print(f"[skip] Unknown shadow: {sh}")
            continue

        score_fn = make_local_score(reward_fn, sp)
        w_fn     = make_weight_fn(reward_fn, sp)

        for disp in dispatches:
            t0 = time.perf_counter()
            if disp == "greedy":
                res = simulate(theta, score_fn, reward_fn, decision_rule="naive",
                               timestamps=timestamps, time_window=args.d,
                               policy="score", weight_fn=None, shadow=None, seed=args.seed)
            elif disp == "greedy+":
                res = simulate(theta, score_fn, reward_fn, decision_rule="threshold",
                               timestamps=timestamps, time_window=args.d,
                               policy="score", weight_fn=None, shadow=None, seed=args.seed)
            elif disp == "batch":
                res = simulate(theta, score_fn, reward_fn, decision_rule="policy",
                               timestamps=timestamps, time_window=args.d,
                               policy="batch", weight_fn=w_fn, shadow=sp, seed=args.seed)
            elif disp == "rbatch":
                res = simulate(theta, score_fn, reward_fn, decision_rule="policy",
                               timestamps=timestamps, time_window=args.d,
                               policy="rbatch", weight_fn=w_fn, shadow=sp, seed=args.seed)
            else:
                print(f"[skip] Unknown dispatch: {disp}")
                continue
            run_time = time.perf_counter() - t0

            r = res["total_savings"]; pooled_pct = res["pooled_pct"]
            ratio_lp = (r / lp_total) if lp_total > 0 else float("nan")
            gap_lp   = _safe_gap(lp_total, r)
            if opt_total is not None:
                ratio_opt = (r / opt_total) if opt_total > 0 else float("nan")
                gap_opt   = _safe_gap(opt_total, r)
                ratio_opt_str = f"{ratio_opt:5.2f}x"; gap_opt_str = f"{gap_opt:9.3f}"
            else:
                ratio_opt_str = "  n/a";          gap_opt_str = "     n/a"

            print(f"{sh.upper():<10} {disp:<12} {pooled_pct:7.1f}% {r:9.3f}  "
                  f"{ratio_lp:5.2f}x {gap_lp:9.3f}  {ratio_opt_str} {gap_opt_str}  "
                  f"{len(res['pairs']):>6}  {len(res['solos']):>6}  {run_time:7.3f}")

            rows.append({
                "shadow": sh, "dispatch": disp, "n": args.n, "d": args.d, "seed": args.seed,
                "savings": r, "pooled_pct": pooled_pct,
                "ratio_lp": ratio_lp, "lp_gap": gap_lp,
                "ratio_opt": (None if opt_total is None else (r / opt_total if opt_total > 0 else None)),
                "opt_gap":   (None if opt_total is None else _safe_gap(opt_total, r)),
                "pairs": len(res["pairs"]), "solos": len(res["solos"]),
                "time_s": run_time, "method": ("score" if "greedy" in disp else disp)
            })

    _write_csv(rows, args.save_csv)

if __name__ == "__main__":
    main()
