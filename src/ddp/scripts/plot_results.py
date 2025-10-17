#!/usr/bin/env python3
"""Unified plotting entry-point for Dynamic Delivery Pooling experiment results.

This script supersedes ``plot_many.py`` (shadow × dispatch grids) and
``plot_sweep.py`` (parameter sweeps).  It accepts either a per-trial CSV produced
by :mod:`ddp.scripts.run_many`/``sweep_param`` or a pre-aggregated CSV produced
by :mod:`ddp.scripts.aggregate_results`.

Examples
--------
Generate heatmaps for all available metrics at ``d=25``::

    python -m ddp.scripts.plot_results --csv_agg results_agg.csv --mode grid \
        --d 25 --outdir figs --metric all

Plot mean savings over a sweep parameter for a subset of policies::

    python -m ddp.scripts.plot_results --csv_full sweep_trials.csv --mode sweep \
        --metric mean_savings --include_policies "naive+greedy,pb+greedy" \
        --out figs/sweep_savings.png
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from ddp.scripts import aggregate_results


# ---------------------------------------------------------------------------
# Data loading & cleaning helpers


@dataclass(frozen=True)
class LoadedData:
    """Container bundling the aggregated DataFrame and its source description."""

    df: pd.DataFrame
    source: str


def _clean_category(series: pd.Series, *, lower: bool = False) -> pd.Series:
    """Strip whitespace, optionally lowercase, while preserving missing values."""

    def _clean(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return pd.NA
            return text.lower() if lower else text
        return value

    return series.map(_clean)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _load_data(csv_full: str | None, csv_agg: str | None) -> LoadedData:
    if csv_full:
        df = aggregate_results.aggregate(csv_full)
        source = f"aggregated from trials: {csv_full}"
    elif csv_agg:
        csv_path = Path(csv_agg)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        source = f"pre-aggregated CSV: {csv_agg}"
    else:
        raise ValueError("Either --csv_full or --csv_agg must be provided")

    # Harmonise categorical columns.
    for col, lower in [("shadow", True), ("dispatch", True), ("param", True)]:
        if col in df.columns:
            df[col] = _clean_category(df[col], lower=lower)

    # Ensure numeric columns are numeric for consistent filtering.
    numeric_candidates: set[str] = set(aggregate_results.NUMERIC_FIELDS)
    numeric_candidates.update(["n", "d", "param_value", "trial_count"])
    _coerce_numeric(df, numeric_candidates)

    return LoadedData(df=df, source=source)


def _parse_metric_list(metric_arg: str | None, available: Sequence[str]) -> list[str]:
    """Return the metrics requested by the user, validating availability."""

    if not available:
        raise ValueError("No metric columns available in aggregated data")

    if not metric_arg or metric_arg.strip().lower() == "all":
        return list(available)

    metrics: list[str] = []
    for token in metric_arg.split(","):
        name = token.strip()
        if not name:
            continue
        if name not in available:
            raise ValueError(
                f"Requested metric '{name}' not present. Available metrics: {', '.join(available)}"
            )
        metrics.append(name)

    if not metrics:
        raise ValueError("No valid metrics parsed from --metric argument")

    return metrics


def _policy_key(row: pd.Series) -> str:
    shadow = row.get("shadow", pd.NA)
    dispatch = row.get("dispatch", pd.NA)
    return f"{shadow}+{dispatch}"


_METRIC_LABELS: dict[str, str] = {
    "savings": "total reward",
    "pooled_pct": "match rate",
    "ratio_lp": "LP Ratio",
    "ratio_opt": "Ratio",
    "time-s": "Running Time (s)",
}


_PARAM_LABELS: dict[str, str] = {
    "d": "Time window (s)",
}


_SHADOW_COLOR_FAMILIES: dict[str, tuple[str, ...]] = {
    # Keep policy colours stable across plots for recognisable comparisons.
    "opt": ("#111111",),
    "naive": ("#f59e0b",),
    "pb": ("#2563eb",),
    "hd": ("#16a34a",),
    "ad": ("#d946ef",),
}


@dataclass(frozen=True)
class _DispatchStyle:
    linestyle: str
    color_factor: float = 1.0
    marker_scale: float = 1.0


_DISPATCH_STYLES: dict[str, _DispatchStyle] = {
    "greedy": _DispatchStyle(linestyle="-", color_factor=1.0, marker_scale=1.0),
    "batch": _DispatchStyle(linestyle=":", color_factor=0.55, marker_scale=1.55),
    "rbatch": _DispatchStyle(linestyle="--", color_factor=0.75, marker_scale=1.25),
    "opt": _DispatchStyle(linestyle="-", color_factor=1.0, marker_scale=1.0),
}


_FALLBACK_DISPATCH_STYLES: tuple[_DispatchStyle, ...] = (
    _DispatchStyle(linestyle="-", color_factor=1.0, marker_scale=1.0),
    _DispatchStyle(linestyle="--", color_factor=1.0, marker_scale=1.0),
    _DispatchStyle(linestyle="-.", color_factor=1.0, marker_scale=1.0),
    _DispatchStyle(linestyle=":", color_factor=1.0, marker_scale=1.0),
)


_UNKNOWN_DISPATCH_CACHE: dict[str, _DispatchStyle] = {}


_SHADOW_MARKERS: dict[str, str] = {
    "naive": "o",
    "pb": "^",
    "hd": "s",
    "ad": "x",
    "opt": "D",
}


_BASE_MARKERSIZE = 6.5


_DISPATCH_ORDER: tuple[str, ...] = ("greedy", "batch", "rbatch", "opt")


_SHADOW_ORDER: tuple[str, ...] = ("naive", "pb", "hd", "ad", "opt")


_POLICY_SHADOW_LABELS: dict[str, str] = {
    "opt": "OPT",
    "naive": "NAIVE",
    "pb": "PB",
    "hd": "HD",
    "ad": "AD",
}


_POLICY_DISPATCH_LABELS: dict[str, str] = {
    "batch": "BAT",
    "rbatch": "RBAT",
    "greedy": "GRE",
    "opt": "OPT",
}


def _format_metric_label(metric: str) -> str:
    base = metric
    if base.startswith("mean_"):
        base = base[len("mean_") :]
    label = _METRIC_LABELS.get(base)
    if label:
        return label
    return base.replace("_", " ")


def _format_param_label(param: str | None) -> str:
    if not param:
        return "param_value"
    return _PARAM_LABELS.get(param, param)


def _split_policy_key(policy: str) -> tuple[str, str]:
    if "+" in policy:
        shadow, dispatch = policy.split("+", 1)
        return shadow, dispatch
    return policy, ""


def _shadow_family(shadow: str) -> str | None:
    lowered = shadow.lower()
    for family in _SHADOW_COLOR_FAMILIES:
        if lowered == family:
            return family
        if lowered.startswith(f"{family}+"):
            return family
        if lowered.startswith(f"{family}-"):
            return family
        if lowered.startswith(f"{family}_"):
            return family
    return None


def _assign_shadow_colors(shadows: Iterable[str]) -> dict[str, str]:
    unique = sorted({s for s in shadows if isinstance(s, str) and s})
    color_map: dict[str, str] = {}

    fallback_colors: list[str] = []
    color_cycle = plt.rcParams.get("axes.prop_cycle")
    if color_cycle:
        fallback_colors = color_cycle.by_key().get("color", []) or []
    if not fallback_colors:
        cmap = plt.get_cmap("tab20", max(len(unique), 1))
        fallback_colors = [cmap(i) for i in range(len(unique))]

    fallback_iter = iter(fallback_colors)

    for shadow in unique:
        family = _shadow_family(shadow)
        if family and family in _SHADOW_COLOR_FAMILIES:
            color_map[shadow] = _SHADOW_COLOR_FAMILIES[family][0]
        else:
            try:
                color_map[shadow] = next(fallback_iter)
            except StopIteration:
                cmap = plt.get_cmap("tab20", max(len(unique), 1))
                color_map[shadow] = cmap(len(color_map) % max(len(unique), 1))

    return color_map


def _shadow_marker(shadow: str) -> str:
    family = _shadow_family(shadow)
    if family and family in _SHADOW_MARKERS:
        return _SHADOW_MARKERS[family]
    return _SHADOW_MARKERS.get(shadow.lower(), "o")


def _scale_color(color: str | tuple[float, float, float] | tuple[float, float, float, float], factor: float) -> tuple[float, float, float, float]:
    rgba = mcolors.to_rgba(color)
    r, g, b, a = rgba
    r = min(max(r * factor, 0.0), 1.0)
    g = min(max(g * factor, 0.0), 1.0)
    b = min(max(b * factor, 0.0), 1.0)
    return (r, g, b, a)


def _dispatch_style(dispatch: str) -> _DispatchStyle:
    key = dispatch.lower() if isinstance(dispatch, str) else ""
    style = _DISPATCH_STYLES.get(key)
    if style:
        return style

    cached = _UNKNOWN_DISPATCH_CACHE.get(key)
    if cached:
        return cached

    idx = len(_UNKNOWN_DISPATCH_CACHE) % len(_FALLBACK_DISPATCH_STYLES)
    style = _FALLBACK_DISPATCH_STYLES[idx]
    _UNKNOWN_DISPATCH_CACHE[key] = style
    return style


def _policy_sort_key(policy: str) -> tuple[int, int, str]:
    shadow, dispatch = _split_policy_key(policy)
    shadow_family = _shadow_family(shadow) or (shadow.lower() if isinstance(shadow, str) else "")
    dispatch_key = dispatch.lower() if isinstance(dispatch, str) else ""

    dispatch_idx = _DISPATCH_ORDER.index(dispatch_key) if dispatch_key in _DISPATCH_ORDER else len(_DISPATCH_ORDER)
    shadow_idx = _SHADOW_ORDER.index(shadow_family) if shadow_family in _SHADOW_ORDER else len(_SHADOW_ORDER)
    return (dispatch_idx, shadow_idx, policy)


def _format_policy_label(policy: str) -> str:
    shadow, dispatch = _split_policy_key(policy)
    shadow_key = shadow.lower() if isinstance(shadow, str) else ""
    dispatch_key = dispatch.lower() if isinstance(dispatch, str) else ""

    if shadow_key == "opt" and dispatch_key == "opt":
        return "OPT"

    def _format_shadow(name: str) -> str:
        if not name:
            return ""
        key = name.lower()
        mapped = _POLICY_SHADOW_LABELS.get(key)
        if mapped:
            return mapped
        return name.upper().replace("_", " ")

    def _format_dispatch(name: str) -> str:
        if not name:
            return ""
        key = name.lower()
        mapped = _POLICY_DISPATCH_LABELS.get(key)
        if mapped:
            return mapped
        return name.upper().replace("_", " ")

    shadow_label = _format_shadow(shadow)
    dispatch_label = _format_dispatch(dispatch)

    if shadow_label and dispatch_label:
        return f"{shadow_label} + {dispatch_label}"
    return shadow_label or dispatch_label or policy


def _build_style_mappings(policies: Iterable[str]) -> dict[str, str]:
    shadows = [shadow for shadow, _ in [_split_policy_key(p) for p in policies]]
    color_map = _assign_shadow_colors(shadows)
    for shadow in shadows:
        if shadow not in color_map and isinstance(shadow, str) and shadow:
            color_map[shadow] = "#636363"
    return color_map


def _parse_policy_filter(text: str) -> set[str] | None:
    if not text or not text.strip():
        return None
    policies = {item.strip() for item in text.split(",") if item.strip()}
    return policies or None


def _filter_by_policy(df: pd.DataFrame, include: set[str] | None) -> pd.DataFrame:
    if not include:
        return df
    mask = df.apply(lambda row: _policy_key(row) in include, axis=1)
    return df.loc[mask].reset_index(drop=True)


def _maybe_filter_categorical(df: pd.DataFrame, column: str, value: str | None) -> pd.DataFrame:
    if value is None or column not in df.columns:
        return df
    return df.loc[df[column] == value].reset_index(drop=True)


def _maybe_filter_numeric(
    df: pd.DataFrame, column: str, value: float | None, *, tol: float = 1e-9
) -> pd.DataFrame:
    if value is None or column not in df.columns:
        return df
    series = pd.to_numeric(df[column], errors="coerce")
    mask = series.sub(value).abs() <= tol
    return df.loc[mask].reset_index(drop=True)


def _unique_or_warn(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns:
        return None

    series = pd.Series(df[column].dropna())
    if series.empty:
        return None

    if series.nunique(dropna=True) == 1:
        value = series.iloc[0]
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):g}"
        return str(value)

    if pd.api.types.is_numeric_dtype(series):
        sorted_vals = np.sort(series.astype(float).to_numpy())
        return f"{sorted_vals[0]:g}..{sorted_vals[-1]:g}"

    return None


# ---------------------------------------------------------------------------
# Grid plotting (shadow × dispatch heatmap)


def _plot_metric_grid(
    df: pd.DataFrame,
    metric: str,
    std_col: str | None,
    *,
    outdir: Path,
    n_label: str | None,
    d_label: str | None,
    param_label: str | None,
) -> Path:
    pivot = df.pivot_table(index="shadow", columns="dispatch", values=metric, aggfunc="mean")
    if pivot.empty:
        raise ValueError(f"No data available to plot metric '{metric}'.")

    shadows = list(pivot.index)
    dispatches = list(pivot.columns)
    vals = pivot.to_numpy(dtype=float)

    stds: np.ndarray | None = None
    if std_col and std_col in df.columns:
        std_pivot = df.pivot_table(index="shadow", columns="dispatch", values=std_col, aggfunc="mean")
        stds = std_pivot.reindex(index=shadows, columns=dispatches).to_numpy(dtype=float)

    fig_w = 1.6 * max(3, len(dispatches)) + 1.5
    fig_h = 1.2 * max(3, len(shadows)) + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked_vals = np.ma.masked_invalid(vals)
    im = ax.imshow(masked_vals, aspect="auto")

    ax.set_xticks(np.arange(len(dispatches)))
    ax.set_xticklabels(dispatches)
    ax.set_yticks(np.arange(len(shadows)))
    ax.set_yticklabels(shadows)

    ax.set_xticks(np.arange(-0.5, len(dispatches), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(shadows), 1), minor=True)
    ax.grid(which="minor", linewidth=0.5, color="white", alpha=0.6)
    ax.grid(which="major", linewidth=0.8, color="white", alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i, shadow in enumerate(shadows):
        for j, dispatch in enumerate(dispatches):
            mean_val = vals[i, j]
            if math.isnan(mean_val):
                continue
            if stds is not None and not math.isnan(stds[i, j]):
                text = f"{mean_val:.3g}\n±{stds[i, j]:.3g}"
            else:
                text = f"{mean_val:.3g}"
            ax.text(j, i, text, ha="center", va="center")

    ax.set_xlabel("dispatch")
    ax.set_ylabel("shadow")

    metric_label = metric.replace("mean_", "").replace("_", " ")
    title_parts = [metric_label]
    if n_label:
        title_parts.append(f"n={n_label}")
    if d_label:
        title_parts.append(f"d={d_label}")
    if param_label:
        title_parts.append(param_label)
    ax.set_title(" | ".join(title_parts))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    safe_metric = metric_label.replace(" ", "_").replace("/", "-")
    out_path = outdir / f"{safe_metric}_grid.png"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _run_grid_mode(args: argparse.Namespace, data: LoadedData) -> None:
    df = data.df.copy()

    if args.include_policies:
        include = _parse_policy_filter(args.include_policies)
        df = _filter_by_policy(df, include)

    df = _maybe_filter_categorical(df, "param", args.param)
    df = _maybe_filter_numeric(df, "param_value", args.param_value)
    df = _maybe_filter_numeric(df, "n", args.n)
    df = _maybe_filter_numeric(df, "d", args.d)

    if df.empty:
        raise ValueError("No rows remain after applying filters for grid mode.")

    available_metrics = [col for col in df.columns if col.startswith("mean_")]
    metrics = _parse_metric_list(args.metric, available_metrics)

    std_for = {metric: f"std_{metric[len('mean_') :]}" for metric in metrics}

    outdir = Path(args.outdir or "figs")

    n_label = _unique_or_warn(df, "n")
    d_label = _unique_or_warn(df, "d")
    param_label = None
    if "param" in df.columns:
        param_values = df["param"].dropna().unique()
        if len(param_values) == 1:
            param_name = str(param_values[0])
            if "param_value" in df.columns:
                val = _unique_or_warn(df, "param_value")
                if val is not None:
                    param_label = f"{param_name}={val}"
            else:
                param_label = param_name

    saved: list[Path] = []
    for metric in metrics:
        std_col = std_for.get(metric)
        saved.append(
            _plot_metric_grid(
                df,
                metric,
                std_col if std_col in df.columns else None,
                outdir=outdir,
                n_label=n_label,
                d_label=d_label,
                param_label=param_label,
            )
        )

    print(f"Loaded data ({data.source})")
    print(f"Saved {len(saved)} grid figure(s) to {outdir.resolve()}")
    for path in saved:
        print(f" - {path}")


# ---------------------------------------------------------------------------
# Sweep plotting (metric vs param_value lines)


def _compute_out_path(csv_hint: str | None, metric: str, out_arg: str | None) -> Path:
    metric_slug = metric.replace(" ", "_")
    if out_arg:
        out_path = Path(out_arg)
        if out_path.is_dir() or out_arg.endswith(os.sep) or out_path.suffix == "":
            base = Path(csv_hint or "results").stem
            return out_path / f"{base}_{metric_slug}.png"
        root, ext = os.path.splitext(out_arg)
        ext = ext or ".png"
        return Path(f"{root}_{metric_slug}{ext}")
    base = Path(csv_hint or "results").stem
    return Path("figs") / f"{base}_{metric_slug}.png"


def _plot_metric_sweep(
    df: pd.DataFrame,
    metric: str,
    include: set[str] | None,
    *,
    title: str | None,
    csv_hint: str | None,
    out_arg: str | None,
    show_legend: bool = True,
) -> Path:
    std_col = None
    if metric.startswith("mean_"):
        std_candidate = f"std_{metric[len('mean_') :]}"
        if std_candidate in df.columns:
            std_col = std_candidate

    metric_base = metric[len("mean_") :] if metric.startswith("mean_") else metric

    by_policy: dict[str, list[pd.Series]] = defaultdict(list)
    for _, row in df.iterrows():
        policy = _policy_key(row)
        if include and policy not in include:
            continue
        by_policy[policy].append(row)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    policies = sorted(by_policy.keys(), key=_policy_sort_key)
    color_map = _build_style_mappings(policies)

    drew_any = False
    log_candidates_x: list[np.ndarray] = []
    log_candidates_y: list[np.ndarray] = []
    for policy in policies:
        rows = by_policy[policy]
        policy_df = pd.DataFrame(rows)
        xs = pd.to_numeric(policy_df.get("param_value"), errors="coerce")
        ys = pd.to_numeric(policy_df.get(metric), errors="coerce")
        if xs.isna().all() or ys.isna().all():
            continue

        order = np.argsort(xs.to_numpy())
        xs = xs.to_numpy()[order]
        ys = ys.to_numpy()[order]
        mask = ~np.isnan(xs) & ~np.isnan(ys)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size == 0:
            continue

        if std_col:
            std_values = (
                pd.to_numeric(policy_df.get(std_col), errors="coerce")
                .to_numpy()
            )
            std_values = std_values[order]

            if "trial_count" in policy_df.columns:
                counts = (
                    pd.to_numeric(policy_df.get("trial_count"), errors="coerce")
                    .to_numpy()
                )
                counts = counts[order]
            else:
                counts = np.full(std_values.shape, np.nan, dtype=float)

            std_values = std_values[mask]
            counts = counts[mask]

            if counts.size:
                counts = counts.astype(float, copy=False)
                sem = np.full_like(std_values, np.nan, dtype=float)
                valid = (counts > 0) & np.isfinite(std_values)
                sem[valid] = std_values[valid] / np.sqrt(counts[valid])
                yerr = 2.0 * sem
            else:
                yerr = np.full_like(std_values, np.nan, dtype=float)
        else:
            yerr = None

        shadow, dispatch = _split_policy_key(policy)
        label = _format_policy_label(policy)
        shadow_family = _shadow_family(shadow) if isinstance(shadow, str) else None
        is_hd_shadow = shadow_family == "hd"

        if shadow == "opt" and dispatch == "opt":
            marker = _SHADOW_MARKERS["opt"]
            linestyle = _DISPATCH_STYLES["opt"].linestyle
            final_color = mcolors.to_rgba(_SHADOW_COLOR_FAMILIES["opt"][0])
            markersize = _BASE_MARKERSIZE * _DISPATCH_STYLES["opt"].marker_scale
        else:
            base_color = color_map.get(shadow, "#636363")
            dispatch_style = _dispatch_style(dispatch)
            final_color = _scale_color(base_color, dispatch_style.color_factor)
            marker = None if is_hd_shadow else _shadow_marker(shadow)
            linestyle = "-." if is_hd_shadow else dispatch_style.linestyle
            markersize = _BASE_MARKERSIZE * dispatch_style.marker_scale

        if is_hd_shadow:
            yerr = None
            ax.plot(
                xs,
                ys,
                color=final_color,
                linestyle=linestyle,
                marker=marker,
                label=label,
            )
        else:
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                color=final_color,
                marker=marker,
                linestyle=linestyle,
                markersize=markersize,
                markerfacecolor=final_color,
                markeredgecolor=final_color,
                capsize=3,
                label=label,
            )
        drew_any = True

        if metric_base == "time-s":
            log_candidates_x.append(xs)
            log_candidates_y.append(ys)

    if not drew_any:
        raise ValueError(f"No sweep data available to plot metric '{metric}'.")

    param_name = df.get("param", pd.Series(dtype=object)).dropna().unique()
    if len(param_name) == 1:
        x_label = _format_param_label(str(param_name[0]))
    else:
        x_label = "param_value"
    ax.set_xlabel(x_label)

    y_label = _format_metric_label(metric)
    ax.set_ylabel(y_label)

    if "ratio" in metric_base.lower():
        ax.set_ylim(top=1.0)

    if metric_base == "time-s" and log_candidates_x and log_candidates_y:
        all_x = np.concatenate(log_candidates_x)
        all_y = np.concatenate(log_candidates_y)
        if np.all(all_x > 0) and np.all(all_y > 0):
            ax.set_xscale("log")
            ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)

    if drew_any and show_legend:
        ncol = max(1, min(4, len(policies)))
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.22),
            borderaxespad=0,
            fontsize=9,
            frameon=False,
            ncol=ncol,
        )

    fig.tight_layout(pad=0.8)

    out_path = _compute_out_path(csv_hint, metric, out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _run_sweep_mode(args: argparse.Namespace, data: LoadedData) -> None:
    df = data.df.copy()

    df = _maybe_filter_categorical(df, "param", args.param)
    df = _maybe_filter_numeric(df, "n", args.n)
    df = _maybe_filter_numeric(df, "d", args.d)

    if "param_value" not in df.columns:
        raise ValueError("Aggregated data does not contain 'param_value' required for sweep plots.")

    include = _parse_policy_filter(args.include_policies)

    available_metrics = [col for col in df.columns if col.startswith("mean_")]
    metrics = _parse_metric_list(args.metric, available_metrics)

    saved: list[Path] = []
    for metric in metrics:
        saved.append(
            _plot_metric_sweep(
                df,
                metric,
                include,
                title=args.title,
                csv_hint=args.csv_agg or args.csv_full,
                out_arg=args.out,
                show_legend=not args.no_legend,
            )
        )

    print(f"Loaded data ({data.source})")
    print(f"Saved {len(saved)} sweep figure(s)")
    for path in saved:
        print(f" - {path}")


# ---------------------------------------------------------------------------
# CLI entry point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot aggregated Dynamic Delivery Pooling metrics (grid or sweep modes)."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv_full",
        help="Path to per-trial CSV results. Aggregation is performed automatically.",
    )
    source.add_argument(
        "--csv_agg",
        "--csv",
        dest="csv_agg",
        help="Path to pre-aggregated CSV (alias: --csv).",
    )

    parser.add_argument(
        "--mode",
        choices=["grid", "sweep"],
        required=True,
        help="Plot style: 'grid' for shadow×dispatch heatmaps, 'sweep' for parameter trends.",
    )
    parser.add_argument(
        "--metric",
        default="all",
        help="Metric name or comma-separated list to plot (use 'all' for every mean_* metric).",
    )
    parser.add_argument(
        "--include_policies",
        default="",
        help="Comma-separated list of shadow+dispatch policy keys to include (default: all).",
    )
    parser.add_argument("--param", help="Filter rows by parameter name before plotting.")
    parser.add_argument(
        "--param_value",
        type=float,
        help="Filter rows by parameter value before plotting (numeric comparison).",
    )
    parser.add_argument("--n", type=float, help="Filter rows by fleet size n before plotting.")
    parser.add_argument("--d", type=float, help="Filter rows by demand rate d before plotting.")

    parser.add_argument("--outdir", default="figs", help="Output directory for grid mode figures.")
    parser.add_argument(
        "--out",
        default="",
        help="Output path or directory for sweep mode (mirrors legacy plot_sweep behaviour).",
    )
    parser.add_argument("--title", default="", help="Custom title for sweep mode (single metric).")
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide legends in sweep mode (useful when plotting many policies).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    data = _load_data(args.csv_full, args.csv_agg)

    if args.mode == "grid":
        _run_grid_mode(args, data)
    else:
        _run_sweep_mode(args, data)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
