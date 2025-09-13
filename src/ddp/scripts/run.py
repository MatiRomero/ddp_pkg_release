import time, csv, numpy as np
from ddp.engine.sim import simulate
from ddp.engine.opt import compute_opt, compute_lp_relaxation
from ddp.algorithms.potential import potential as potential_vec


def reward_fn(i, j, theta):
    # Toy 1D distance-savings
    return min(theta[i], theta[j])


def make_local_score(reward_fn, sp):
    # local greedy score: r(i,j) - s_j
    def score(i, j, theta):
        return reward_fn(i, j, theta) - float(sp[j])
    return score


def make_weight_fn(reward_fn, sp):
    # (r)batch weight: r(i,j) - s_i - s_j
    def w(i, j, theta):
        return reward_fn(i, j, theta) - float(sp[i]) - float(sp[j])
    return w


def _safe_gap(upper, val):
    return max(upper - val, 0.0)


def _write_csv(rows, path):
    if not path:
        return
    fields = [
        "shadow",
        "dispatch",
        "n",
        "d",
        "seed",
        "savings",
        "pooled_pct",
        "ratio_lp",
        "lp_gap",
        "ratio_opt",
        "opt_gap",
        "pairs",
        "solos",
        "time_s",
        "method",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_instance(
    theta,
    timestamps,
    d,
    shadows=("naive", "pb", "hd"),
    dispatches=("greedy", "greedy+", "batch", "rbatch"),
    seed=0,
    with_opt=False,
    opt_method="auto",
    save_csv="",
    print_table=True,
    return_details=False,
    print_matches=False,
):
    """Run the SHADOW × DISPATCH grid on a given instance.

    Parameters
    ----------
    theta : 1D array-like
    timestamps : 1D array-like (same length as theta)
    d : float or 1D array-like (time window)
    return_details : bool
        If True, return per-(shadow,dispatch) final matches/solos for downstream debugging/plots.
    print_matches : bool
        If True, print final matches/solos per algorithm after the table.

    Returns
    -------
    dict with keys: rows, lp_total, lp_time, opt_total, opt_time, opt_pairs, opt_method,
    and if return_details: details, plus instance arrays (theta, timestamps, due_time).
    """
    theta = np.asarray(theta, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    n = len(theta)

    # compute due_time for plotting/debug (not used by simulate)
    if np.isscalar(d):
        due_time = timestamps + float(d)
    else:
        d_arr = np.asarray(d, dtype=float)
        assert len(d_arr) == n, "len(time_window) must equal len(theta)"
        due_time = timestamps + d_arr

    # LP (upper bound + duals for HD) — compute once
    t0 = time.perf_counter()
    lp = compute_lp_relaxation(theta, reward_fn)
    lp_time = time.perf_counter() - t0
    lp_total, duals = float(lp["total_upper"]), np.array(lp["duals"], dtype=float)

    # Optional OPT once
    opt_total = opt_pairs = opt_m = None
    opt_time = 0.0
    if with_opt:
        t0 = time.perf_counter()
        opt = compute_opt(theta, reward_fn, method=opt_method)
        opt_time = time.perf_counter() - t0
        opt_total, opt_pairs, opt_m = float(opt["total_reward"]), opt["pairs"], opt["method"]

    if print_table:
        print(f"LP_RELAX  upper={lp_total:.3f}  method={lp['method']}  time={lp_time:.3f}s")
        if with_opt:
            print(
                f"OPT       total={opt_total:.3f}  pairs={len(opt_pairs):>4}  method={opt_m}  time={opt_time:.3f}s"
            )
        print(f"\nInstance: n={n}, d={d}")
        print("-" * 142)
        print(
            "SHADOW     DISPATCH     POOLED%   SAVINGS    R/LP   LP_GAP    R/OPT  OPT_GAP   #PAIRS  #SOLOS   TIME(s)"
        )

    rows = []
    details = {}  # (shadow, dispatch) -> {pairs:[(i,j),...], solos:[i,...]}

    for sh in shadows:
        # shadow vector
        if sh == "naive":
            sp = np.zeros(n, dtype=float)
        elif sh == "pb":
            sp = potential_vec(theta)
        elif sh == "hd":
            sp = duals
        else:
            if print_table:
                print(f"[skip] Unknown shadow: {sh}")
            continue

        score_fn = make_local_score(reward_fn, sp)
        w_fn = make_weight_fn(reward_fn, sp)

        for disp in dispatches:
            t_run = time.perf_counter()
            if disp == "greedy":
                res = simulate(
                    theta,
                    score_fn,
                    reward_fn,
                    "naive",
                    timestamps,
                    d,
                    policy="score",
                    weight_fn=None,
                    shadow=None,
                    seed=seed,
                )
            elif disp == "greedy+":
                res = simulate(
                    theta,
                    score_fn,
                    reward_fn,
                    "threshold",
                    timestamps,
                    d,
                    policy="score",
                    weight_fn=None,
                    shadow=None,
                    seed=seed,
                )
            elif disp == "batch":
                res = simulate(
                    theta,
                    score_fn,
                    reward_fn,
                    "policy",
                    timestamps,
                    d,
                    policy="batch",
                    weight_fn=w_fn,
                    shadow=sp,
                    seed=seed,
                )
            elif disp == "rbatch":
                res = simulate(
                    theta,
                    score_fn,
                    reward_fn,
                    "policy",
                    timestamps,
                    d,
                    policy="rbatch",
                    weight_fn=w_fn,
                    shadow=sp,
                    seed=seed,
                )
            else:
                if print_table:
                    print(f"[skip] Unknown dispatch: {disp}")
                continue
            run_time = time.perf_counter() - t_run

            r = res["total_savings"]
            pooled_pct = res["pooled_pct"]
            ratio_lp = (r / lp_total) if lp_total > 0 else float("nan")
            gap_lp = _safe_gap(lp_total, r)
            if with_opt and opt_total is not None:
                ratio_opt = (r / opt_total) if opt_total > 0 else float("nan")
                gap_opt = _safe_gap(opt_total, r)
                ratio_opt_str = f"{ratio_opt:5.2f}x"
                gap_opt_str = f"{gap_opt:9.3f}"
            else:
                ratio_opt_str = "  n/a"
                gap_opt_str = "     n/a"

            if print_table:
                print(
                    f"{sh.upper():<10} {disp:<12} {pooled_pct:7.1f}% {r:9.3f}  "
                    f"{ratio_lp:5.2f}x {gap_lp:9.3f}  {ratio_opt_str} {gap_opt_str}  "
                    f"{len(res['pairs']):>6}  {len(res['solos']):>6}  {run_time:7.3f}"
                )

            rows.append(
                {
                    "shadow": sh,
                    "dispatch": disp,
                    "n": n,
                    "d": d if np.isscalar(d) else None,
                    "seed": seed,
                    "savings": r,
                    "pooled_pct": pooled_pct,
                    "ratio_lp": ratio_lp,
                    "lp_gap": gap_lp,
                    "ratio_opt": (r / opt_total) if (with_opt and opt_total and opt_total > 0) else None,
                    "opt_gap": (_safe_gap(opt_total, r) if (with_opt and opt_total is not None) else None),
                    "pairs": len(res["pairs"]),
                    "solos": len(res["solos"]),
                    "time_s": run_time,
                    "method": ("score" if "greedy" in disp else disp),
                }
            )

            if return_details or print_matches:
                pairs_idx = [(i, j) for (i, j, _, _) in res["pairs"]]
                info = {"pairs": pairs_idx, "solos": list(res["solos"]) }
                details[(sh, disp)] = info
                if print_matches:
                    print(
                        f"    -> matches {sh}/{disp}: pairs={pairs_idx}  solos={info['solos']}"
                    )

    _write_csv(rows, save_csv)

    out = {
        "rows": rows,
        "lp_total": lp_total,
        "lp_time": lp_time,
        "opt_total": opt_total,
        "opt_time": opt_time,
        "opt_pairs": opt_pairs,
        "opt_method": opt_m,
        # instance-level arrays for plotting
        "theta": theta,
        "timestamps": timestamps,
        "due_time": due_time,
    }
    if return_details:
        out["details"] = details
    return out


# Optional CLI: run on arrays loaded from .npy files (theta, timestamps) for power users.
def main():
    import argparse

    p = argparse.ArgumentParser(description="Run SHADOW×DISPATCH on a given instance (arrays).")
    p.add_argument("--theta", type=str, help="Path to .npy for theta (1D float array).")
    p.add_argument("--timestamps", type=str, help="Path to .npy for timestamps (1D float array).")
    p.add_argument("--d", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shadows", default="naive,pb,hd")
    p.add_argument("--dispatch", default="greedy,greedy+,batch,rbatch")
    p.add_argument("--with_opt", action="store_true")
    p.add_argument("--opt_method", default="auto", choices=["auto", "networkx", "ilp"])
    p.add_argument("--save_csv", default="")
    p.add_argument("--print_matches", action="store_true")
    p.add_argument("--return_details", action="store_true")
    args = p.parse_args()

    theta = np.load(args.theta) if args.theta else None
    timestamps = np.load(args.timestamps) if args.timestamps else None
    if theta is None:
        raise SystemExit("Provide --theta .npy")
    if timestamps is None:
        timestamps = np.arange(len(theta), dtype=float)

    run_instance(
        theta=theta,
        timestamps=timestamps,
        d=args.d,
        shadows=[s.strip() for s in args.shadows.split(",") if s.strip()],
        dispatches=[d.strip() for d in args.dispatch.split(",") if d.strip()],
        seed=args.seed,
        with_opt=args.with_opt,
        opt_method=args.opt_method,
        save_csv=args.save_csv,
        print_table=True,
        return_details=args.return_details,
        print_matches=args.print_matches,
    )


if __name__ == "__main__":
    main()