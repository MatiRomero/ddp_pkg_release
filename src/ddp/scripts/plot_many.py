#!/usr/bin/env python3
"""
Usage:
  python src/ddp/scripts/plot_many.py --csv results_agg.csv --outdir figs
  # or, if you just drop this file into the repo root:
  python plot_many.py --csv results_agg.csv --outdir figs

This script loads the CSV produced by run_many.py and generates simple bar charts
(with error bars) for key metrics by (shadow, dispatch). No seaborn required.
Each chart is saved as a separate PNG in --outdir.
"""
from __future__ import annotations
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- plotting helpers ---------------------------------
def _ensure_outdir(outdir: str):
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

def _grouped_bar(ax, categories, series_values, series_errors=None, width=0.8):
    """
    Draw grouped bars:
    - categories: list of x-axis categories (e.g., dispatch policies)
    - series_values: dict(series_name -> list of values aligned with categories)
    - series_errors: dict(series_name -> list of errors aligned with categories) or None
    - width: total width allocated per category (0 < width <= 0.9 recommended)
    """
    n_cat = len(categories)
    series_names = list(series_values.keys())
    k = len(series_names)
    bw = width / max(k, 1)
    x = np.arange(n_cat)
    for idx, s in enumerate(series_names):
        vals = series_values[s]
        offs = (idx - (k-1)/2) * bw
        if series_errors is not None and s in series_errors and series_errors[s] is not None:
            ax.bar(x + offs, vals, bw, yerr=series_errors[s], capsize=3, align="center")
        else:
            ax.bar(x + offs, vals, bw, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)

def _savefig(fig, outpath: str):
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# ------------------------------- main logic ----------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["shadow"] = df["shadow"].astype(str).str.strip().str.lower()
    df["dispatch"] = df["dispatch"].astype(str).str.strip().str.lower()
    return df

def plot_metric_grid(df: pd.DataFrame, metric_col: str, std: str | None, outdir: str, n_label: str | None = None, d_label: str | None = None, metric_label: str | None = None):
    """Plot a grid with rows=shadow and cols=dispatch.
    Each cell shows mean and (if available) std over trials, and the background
    is a heatmap of the mean values.
    """
    # Build pivot tables (rows=shadow, cols=dispatch)
    piv_val = df.pivot(index="shadow", columns="dispatch", values=metric_col)
    shadows = list(piv_val.index)
    dispatches = list(piv_val.columns)
    vals = piv_val.values.astype(float)

    stds = None
    if std and std in df.columns:
        piv_std = df.pivot(index="shadow", columns="dispatch", values=std)
        stds = piv_std.values.astype(float)

    # Figure size scales with grid size
    fig_w = 1.6 * max(3, len(dispatches)) + 1.5
    fig_h = 1.2 * max(3, len(shadows)) + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Heatmap of means (default colormap)
    im = ax.imshow(vals)

    # Axis ticks and labels
    ax.set_xticks(np.arange(len(dispatches)))
    ax.set_xticklabels(dispatches)
    ax.set_yticks(np.arange(len(shadows)))
    ax.set_yticklabels(shadows)

    # Draw grid lines between cells
    ax.set_xticks(np.arange(-0.5, len(dispatches), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(shadows), 1), minor=True)
    ax.grid(which='minor', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    # Annotate each cell with mean (and std if available)
    for i in range(len(shadows)):
        for j in range(len(dispatches)):
            m = vals[i, j]
            if stds is not None:
                s = stds[i, j]
                txt = f"{m:.3g}\n±{s:.3g}"
            else:
                txt = f"{m:.3g}"
            ax.text(j, i, txt, ha='center', va='center')

    ax.set_xlabel("dispatch")
    ax.set_ylabel("shadow")
    title_label = metric_label or metric_col
    ax.set_title(f"{title_label} (n={n_label}, d={d_label})")

    # Add colorbar for the means
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    safe_label = (metric_label or metric_col).replace(" ", "_").replace("/", "-")
    out = os.path.join(outdir, f"{safe_label}_grid.png")
    _savefig(fig, out)
    return out

def main():
    ap = argparse.ArgumentParser(description="Plot aggregated DDP results from run_many.py")
    ap.add_argument("--csv", required=True, help="Path to results_agg.csv written by run_many.py")
    ap.add_argument("--outdir", default="figs", help="Directory to save figures (created if missing)")
    ap.add_argument("--metrics", default="ALL",
                    help="Comma-separated list of metrics to plot, or ALL to auto-discover all mean_* columns in the CSV")
    ap.add_argument("--with_std", action="store_true",
                    help="If set, add error bars using std_* columns when present")
    args = ap.parse_args()

    _ensure_outdir(args.outdir)
    df = load_and_clean(args.csv)

    if args.metrics.strip().upper() == "ALL":
        metrics = [col for col in df.columns if col.startswith("mean_")]
    else:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    err_for_metric = {}
    for m in metrics:
        suffix = m[len("mean_"):] if m.startswith("mean_") else None
        err_col = f"std_{suffix}" if suffix else None
        if args.with_std and err_col and err_col in df.columns:
            err_for_metric[m] = err_col
        else:
            err_for_metric[m] = None

    # derive labels for n and d from the CSV (handle mixed values gracefully)
    def _mk_label(col):
        if col not in df.columns:
            return "?"
        vals = pd.unique(df[col])
        try:
            vals = list(np.sort(vals.astype(float)))
        except Exception:
            vals = list(vals)
        if len(vals) == 0:
            return "?"
        if len(vals) == 1:
            return f"{vals[0]:g}" if isinstance(vals[0], (int,float,np.floating)) else str(vals[0])
        return f"{vals[0]:g}..{vals[-1]:g}" if isinstance(vals[0], (int,float,np.floating)) else f"{vals[0]}..{vals[-1]}"
    n_label = _mk_label("n")
    d_label = _mk_label("d")
    saved = []
    for m in metrics:
        err = err_for_metric.get(m)
        label = m.replace("mean_", "").replace("_", " ")
        saved.append(plot_metric_grid(df, m, err, args.outdir, n_label=n_label, d_label=d_label, metric_label=label))

    summary_path = os.path.join(args.outdir, "summary_sorted.csv")
    order_cols = ["shadow", "dispatch", "mean_savings", "mean_pooled_pct", "mean_pairs", "mean_solos"]
    keep_cols = [c for c in order_cols if c in df.columns]
    df.sort_values(keep_cols, ascending=[True, True, False, False, False, True][:len(keep_cols)]).to_csv(summary_path, index=False)

    print("Saved figures:")
    for p in saved:
        print(" -", p)
    print("Saved summary CSV:", summary_path)

if __name__ == "__main__":
    main()