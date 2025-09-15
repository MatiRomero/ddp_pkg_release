# src/ddp/scripts/plot_sweep.py
import argparse, csv, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Set

METRICS_ALL = [
    "mean_savings",
    "mean_pooled_pct",
    "mean_ratio_lp",
    "mean_ratio_opt",
]

STD_FOR = {
    "mean_savings": "std_savings",
    "mean_pooled_pct": "std_pooled_pct",
    "mean_ratio_lp": "std_ratio_lp",
    "mean_ratio_opt": "std_ratio_opt",
}


def _read_csv(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def _policy_key(sh, disp):
    return f"{sh}+{disp}"


def _compute_out_path(csv_path: str, metric: str, out_arg: str) -> str:
    """Figure out where to save the plot for a given metric.
    - If out_arg is empty → figs/<csv_basename>_<metric>.png
    - If out_arg is a directory (existing or endswith sep or no extension) → <out_arg>/<csv_basename>_<metric>.png
    - If out_arg is a file path → add _<metric> before the extension
    """
    base = os.path.splitext(os.path.basename(csv_path))[0]
    default_name = f"{base}_{metric}.png"

    if not out_arg or not out_arg.strip():
        out_path = os.path.join("figs", default_name)
    else:
        # treat as directory if endswith sep, exists as dir, or has no extension
        if out_arg.endswith(os.sep) or os.path.isdir(out_arg) or os.path.splitext(out_arg)[1] == "":
            out_path = os.path.join(out_arg, default_name)
        else:
            root, ext = os.path.splitext(out_arg)
            out_path = f"{root}_{metric}{ext or '.png'}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _plot_metric(ax, rows_by_policy, metric: str, include: Optional[Set[str]]):
    std_col = STD_FOR[metric]
    drew_any = False

    for pol, lst in sorted(rows_by_policy.items()):
        if include and pol not in include:
            continue
        xs = np.array([_safe_float(r.get("param_value", np.nan)) for r in lst])
        ys = np.array([_safe_float(r.get(metric, np.nan)) for r in lst])
        yerr = np.array([_safe_float(r.get(std_col, 0.0)) for r in lst])

        # sort by x
        order = np.argsort(xs)
        xs, ys, yerr = xs[order], ys[order], yerr[order]

        # mask nan
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        xs, ys, yerr = xs[mask], ys[mask], yerr[mask]
        if xs.size == 0:
            continue

        ax.errorbar(xs, ys, yerr=yerr, fmt='-o', capsize=3, label=pol)
        drew_any = True

    return drew_any


def main():
    ap = argparse.ArgumentParser(description="Plot metric(s) vs parameter value from sweep CSV with error bars.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="all",
                    choices=["all"] + METRICS_ALL,
                    help="Which metric to plot. Default: all → one figure per metric.")
    ap.add_argument("--include_policies", default="", help='comma list like "naive+greedy,pb+greedy+" (default: all)')
    ap.add_argument("--title", default="", help="Optional custom title (used for single-metric plots only)")
    ap.add_argument("--out", default="", help="Output path or directory. If omitted, saves to figs/<csv_basename>_<metric>.png")
    args = ap.parse_args()

    rows = _read_csv(args.csv)
    if not rows:
        raise SystemExit("No rows in CSV")

    # Group by policy
    by_pol = defaultdict(list)
    param_name = rows[0].get("param", "param")
    for r in rows:
        pol = _policy_key(r.get("shadow", "?"), r.get("dispatch", "?"))
        by_pol[pol].append(r)

    include = None
    if args.include_policies.strip():
        include = set(p.strip() for p in args.include_policies.split(",") if p.strip())

    metrics_to_plot = METRICS_ALL if args.metric == "all" else [args.metric]

    for metric in metrics_to_plot:
        plt.figure()
        ax = plt.gca()
        any_lines = _plot_metric(ax, by_pol, metric, include)

        ax.set_xlabel(param_name)
        ylab = {
            "mean_savings": "Total savings",
            "mean_pooled_pct": "Pooled %",
            "mean_ratio_lp": "R / LP",
            "mean_ratio_opt": "R / OPT",
        }[metric]
        title = args.title or f"{ylab} vs {param_name}"
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if any_lines:
            ax.legend(loc="best", fontsize=9)
        plt.tight_layout()

        out_path = _compute_out_path(args.csv, metric, args.out)
        plt.savefig(out_path, dpi=160)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()