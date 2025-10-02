"""Plot utilities for visualising :mod:`ddp.scripts.shadow_sweep` CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "geometry",
    "shadow",
    "gamma",
    "tau",
    "metric",
    "mean",
}


def _parse_multi(values: Iterable[str] | None) -> list[str] | None:
    if not values:
        return None
    tokens: list[str] = []
    for value in values:
        chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        tokens.extend(chunks)
    return tokens or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a gamma/tau sweep exported by ddp.scripts.shadow_sweep."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to the aggregated sweep CSV")
    parser.add_argument(
        "--x",
        choices=("gamma", "tau"),
        default="gamma",
        help="Column to use along the x-axis",
    )
    parser.add_argument(
        "--geometry",
        action="append",
        help="Restrict to one or more geometry presets (repeat or comma-separate)",
    )
    parser.add_argument(
        "--shadow",
        action="append",
        help="Restrict to one or more shadow strategies (repeat or comma-separate)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Metric name to plot (defaults to the only metric present)",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional figure title",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to save the rendered figure",
    )
    parser.add_argument(
        "--best-out",
        dest="best_out",
        default="",
        help=(
            "Optional path where the best (gamma, tau) summary CSV should be "
            "written (defaults to <csv>_best.csv)"
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )
    return parser


def _ensure_single_value(df: pd.DataFrame, column: str) -> str:
    unique = sorted({val for val in df[column].dropna().unique()})
    if not unique:
        raise SystemExit(f"Column '{column}' is empty after filtering")
    if len(unique) > 1:
        raise SystemExit(
            f"Multiple values for '{column}' remain ({', '.join(map(str, unique))}); "
            f"please specify --{column.replace('_', '-')}"
        )
    return str(unique[0])


def _format_label(columns: Sequence[str], key: Sequence[str | float]) -> str:
    if not columns:
        return ""
    if len(columns) == 1:
        return f"{columns[0]}={key[0]}"
    return ", ".join(f"{col}={val}" for col, val in zip(columns, key))


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise SystemExit(
            f"CSV is missing required column(s): {missing_list}. "
            "Did you supply the shadow_sweep output?"
        )

    geometry_filter = _parse_multi(args.geometry)
    shadow_filter = _parse_multi(args.shadow)

    if geometry_filter:
        df = df[df["geometry"].isin(geometry_filter)]
    if shadow_filter:
        df = df[df["shadow"].isin(shadow_filter)]

    if args.metric is not None:
        df = df[df["metric"] == args.metric]
        metric = args.metric
    else:
        metric = _ensure_single_value(df, "metric")
        df = df[df["metric"] == metric]

    if df.empty:
        raise SystemExit("No rows remain after applying the requested filters")

    df = df.copy()

    mean_numeric = pd.to_numeric(df["mean"], errors="coerce")
    if mean_numeric.isna().all():
        raise SystemExit("Unable to parse numeric mean values from the CSV")

    best_group_candidates = [
        column
        for column in ("geometry", "d", "resolution", "dispatch", "shadow")
        if column in df.columns
    ]

    best_rows: list[dict[str, object]] = []
    if best_group_candidates:
        grouped_best = df.groupby(best_group_candidates, dropna=False)
    else:
        grouped_best = [((), df)]

    for key, group_df in grouped_best:
        if not isinstance(key, tuple):
            key = (key,)

        group_mean = mean_numeric.loc[group_df.index]
        if group_mean.notna().sum() == 0:
            group_repr = ", ".join(
                f"{column}={value}" for column, value in zip(best_group_candidates, key)
            ) or "overall"
            raise SystemExit(
                "Unable to determine best gamma/tau because all mean values are NaN "
                f"for group {group_repr}"
            )

        best_index = group_mean.idxmax()
        best_entry = df.loc[best_index]

        record: dict[str, object] = {
            "geometry": "",
            "d": "",
            "resolution": "",
            "dispatch": "",
            "shadow": "",
            "best_gamma": best_entry.get("gamma", ""),
            "best_tau": best_entry.get("tau", ""),
            "best_mean": group_mean.loc[best_index],
        }

        for column, value in zip(best_group_candidates, key):
            if pd.isna(value):
                record[column] = ""
            else:
                record[column] = value

        label_parts = [
            f"{column}={record[column]}"
            for column in best_group_candidates
            if record[column] != ""
        ]
        label = ", ".join(label_parts) if label_parts else "overall"
        message = (
            f"Best for {label}: gamma={record['best_gamma']}, "
            f"tau={record['best_tau']}, mean={record['best_mean']}"
        )
        print(message)

        best_rows.append(record)

    best_columns = [
        "geometry",
        "d",
        "resolution",
        "dispatch",
        "shadow",
        "best_gamma",
        "best_tau",
        "best_mean",
    ]

    best_df = pd.DataFrame(best_rows, columns=best_columns)

    best_out_arg = args.best_out.strip()
    if best_out_arg:
        best_out_path = Path(best_out_arg)
    else:
        best_out_path = csv_path.with_name(f"{csv_path.stem}_best.csv")

    best_out_path.parent.mkdir(parents=True, exist_ok=True)
    best_df.sort_values(best_group_candidates or ["geometry"], inplace=True)
    best_df.to_csv(best_out_path, index=False)
    print(f"Best summary written to {best_out_path}")

    x_col = args.x
    plot_columns = [
        col
        for col in ("geometry", "shadow", "dispatch")
        if col in df.columns
    ]
    series_columns = [col for col in plot_columns if df[col].nunique() > 1]

    group_cols = series_columns + [x_col]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(mean_value=("mean", "mean"))
        .reset_index()
    )

    if "std" in df.columns:
        std_values = (
            df.groupby(group_cols, dropna=False)["std"]
            .mean()
            .reset_index(name="std_value")
        )
        grouped = grouped.merge(std_values, on=group_cols, how="left")
    else:
        grouped["std_value"] = float("nan")

    if grouped.empty:
        raise SystemExit("Grouping operation produced no data to plot")

    fig, ax = plt.subplots()

    if series_columns:
        series_iter = grouped.groupby(series_columns, dropna=False)
    else:
        series_iter = [((), grouped)]

    for key, series_df in series_iter:
        if not isinstance(key, tuple):
            key = (key,)
        sorted_df = series_df.sort_values(x_col)
        label = _format_label(series_columns, key)
        ax.plot(sorted_df[x_col], sorted_df["mean_value"], marker="o", label=label)
        if not sorted_df["std_value"].isna().all():
            lower = sorted_df["mean_value"] - sorted_df["std_value"].fillna(0.0)
            upper = sorted_df["mean_value"] + sorted_df["std_value"].fillna(0.0)
            ax.fill_between(sorted_df[x_col], lower, upper, alpha=0.2)

    ax.set_xlabel(x_col)
    ax.set_ylabel(f"mean {metric}")
    title = args.title or f"{metric} vs {x_col}"
    ax.set_title(title)
    if series_columns:
        ax.legend(title=", ".join(series_columns))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
