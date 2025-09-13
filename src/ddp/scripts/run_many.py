import csv, time, argparse
import numpy as np
from ddp.engine.sim import simulate
from ddp.engine.opt import compute_lp_relaxation

def reward_fn(i, j, theta):
    return min(theta[i], theta[j])

def make_local_score(reward_fn, sp):
    def score(i, j, theta):
        return reward_fn(i, j, theta) - float(sp[j])
    return score

def make_weight_fn(reward_fn, sp):
    def w(i, j, theta):
        return reward_fn(i, j, theta) - float(sp[i]) - float(sp[j])
    return w

def run_one(theta, d, seed, shadow, dispatch):
    n = len(theta); timestamps = np.arange(n, dtype=float)
    lp = compute_lp_relaxation(theta, reward_fn)
    if shadow == "naive":
        sp = np.zeros(n, dtype=float)
    elif shadow == "pb":
        sp = theta / 2.0
    elif shadow == "hd":
        sp = np.array(lp["duals"], dtype=float)
    else:
        raise ValueError(f"Unknown shadow: {shadow}")
    score_fn = make_local_score(reward_fn, sp); w_fn = make_weight_fn(reward_fn, sp)

    if dispatch == "greedy":
        res = simulate(theta, score_fn, reward_fn, "naive", timestamps, d, "score", None, None, seed)
    elif dispatch == "greedy+":
        res = simulate(theta, score_fn, reward_fn, "threshold", timestamps, d, "score", None, None, seed)
    elif dispatch == "batch":
        res = simulate(theta, score_fn, reward_fn, "policy", timestamps, d, "batch", w_fn, sp, seed)
    elif dispatch == "rbatch":
        res = simulate(theta, score_fn, reward_fn, "policy", timestamps, d, "rbatch", w_fn, sp, seed)
    else:
        raise ValueError(f"Unknown dispatch: {dispatch}")

    return {"savings": res["total_savings"], "pooled_pct": res["pooled_pct"],
            "pairs": len(res["pairs"]), "solos": len(res["solos"])}

def main():
    ap = argparse.ArgumentParser(description="Aggregate multi-trial stats for SHADOW × DISPATCH")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--d", type=float, default=5.0)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0, help="starting seed (we use seed0..seed0+trials-1)")
    ap.add_argument("--shadows", default="naive,pb,hd")
    ap.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch")
    ap.add_argument("--save_csv", default="results_agg.csv")
    args = ap.parse_args()

    shadows = [s.strip().lower() for s in args.shadows.split(",") if s.strip()]
    dispatches = [d.strip().lower() for d in args.dispatch.split(",") if d.strip()]

    agg = {}; t_start = time.perf_counter()
    for t in range(args.trials):
        seed = args.seed0 + t
        theta = np.random.default_rng(seed).random(args.n)
        for sh in shadows:
            for disp in dispatches:
                key = (sh, disp)
                res = run_one(theta, args.d, seed, sh, disp)
                agg.setdefault(key, {"savings": [], "pooled": [], "pairs": [], "solos": []})
                agg[key]["savings"].append(res["savings"])
                agg[key]["pooled"].append(res["pooled_pct"])
                agg[key]["pairs"].append(res["pairs"])
                agg[key]["solos"].append(res["solos"])

    rows = []
    for (sh, disp), data in agg.items():
        def mean(x): return float(np.mean(x))
        def std(x):  return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        rows.append({"shadow": sh, "dispatch": disp, "n": args.n, "d": args.d,
                     "trials": args.trials, "seed0": args.seed0,
                     "mean_savings": mean(data["savings"]), "std_savings": std(data["savings"]),
                     "mean_pooled_pct": mean(data["pooled"]), "std_pooled_pct": std(data["pooled"]),
                     "mean_pairs": mean(data["pairs"]), "mean_solos": mean(data["solos"])})

    with open(args.save_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    dur = time.perf_counter() - t_start
    print(f"Wrote {args.save_csv} with {len(rows)} rows in {dur:.2f}s")

if __name__ == "__main__":
    main()
