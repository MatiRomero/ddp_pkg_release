"""Compare potentials with hindsight duals in a dataset.

Example
-------
Run the script on a CSV dataset and save the scatter plot::

    python -m ddp.scripts.compare_hd_stats data/hd_dataset.csv --out_plot hd_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


POTENTIAL_COLUMN = "potential"
HINDSIGHT_DUAL_COLUMN = "hindsight_dual"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the dataset CSV produced by build_hd_dataset.py.",
    )
    parser.add_argument(
        "--out_plot",
        type=Path,
        default=None,
        help="Optional destination filename for the scatter plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the scatter plot interactively instead of only saving it.",
    )
    return parser.parse_args()


def _load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        msg = f"CSV dataset not found: {csv_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    missing_cols = {
        column
        for column in (POTENTIAL_COLUMN, HINDSIGHT_DUAL_COLUMN)
        if column not in df.columns
    }
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        msg = f"Dataset missing expected column(s): {missing}"
        raise ValueError(msg)

    return df


def _format_float(value: float) -> str:
    return f"{value: .6g}" if pd.notna(value) else "   nan"


def _print_summary(df: pd.DataFrame) -> None:
    potential = df[POTENTIAL_COLUMN].dropna()
    hindsight = df[HINDSIGHT_DUAL_COLUMN].dropna()

    summary = pd.DataFrame(
        {
            "count": [potential.count(), hindsight.count()],
            "mean": [potential.mean(), hindsight.mean()],
            "median": [potential.median(), hindsight.median()],
            "std": [potential.std(ddof=1), hindsight.std(ddof=1)],
        },
        index=[POTENTIAL_COLUMN, HINDSIGHT_DUAL_COLUMN],
    )

    joined = df[[POTENTIAL_COLUMN, HINDSIGHT_DUAL_COLUMN]].dropna()
    correlation = joined[POTENTIAL_COLUMN].corr(joined[HINDSIGHT_DUAL_COLUMN])
    diff = joined[POTENTIAL_COLUMN] - joined[HINDSIGHT_DUAL_COLUMN]
    mean_abs_diff = diff.abs().mean()
    non_zero = joined[POTENTIAL_COLUMN].replace(0.0, pd.NA).dropna()
    rel_diff = diff.loc[non_zero.index].abs() / non_zero.abs()
    mean_rel_diff = rel_diff.mean()

    print("Summary statistics:")
    with pd.option_context("display.max_columns", None, "display.width", 100):
        print(summary.to_string(float_format=lambda x: f"{x:0.6g}"))

    print("\nPairwise comparison:")
    print(f"Correlation: {_format_float(correlation)}")
    print(f"Mean |potential - hindsight|: {_format_float(mean_abs_diff)}")
    if rel_diff.empty:
        print("Mean relative difference:    nan (no non-zero potentials)")
    else:
        print(f"Mean relative difference: {_format_float(mean_rel_diff)}")


def _plot(df: pd.DataFrame, out_plot: Path | None, show: bool) -> None:
    joined = df[[POTENTIAL_COLUMN, HINDSIGHT_DUAL_COLUMN]].dropna()
    if joined.empty:
        print("No valid data points to plot.")
        return

    potentials = joined[POTENTIAL_COLUMN]
    hindsight = joined[HINDSIGHT_DUAL_COLUMN]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(potentials, hindsight, alpha=0.6, edgecolors="none")

    min_val = float(min(potentials.min(), hindsight.min()))
    max_val = float(max(potentials.max(), hindsight.max()))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.0, label="y = x")

    ax.set_xlabel("Potential")
    ax.set_ylabel("Hindsight dual")
    ax.set_title("Potentials vs. Hindsight duals")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)

    if out_plot is not None:
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, bbox_inches="tight")
        print(f"Scatter plot saved to {out_plot}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = _parse_args()
    df = _load_dataset(Path(args.csv_path))
    _print_summary(df)
    _plot(df, Path(args.out_plot) if args.out_plot else None, bool(args.show))


if __name__ == "__main__":
    main()
