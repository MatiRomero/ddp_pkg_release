#!/usr/bin/env python3
"""Compatibility wrapper that forwards to :mod:`ddp.scripts.plot_results`."""

from __future__ import annotations

import argparse
from typing import Sequence

from ddp.scripts import plot_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot aggregated DDP results as shadow×dispatch heatmaps (legacy wrapper)."
    )
    parser.add_argument("--csv", required=True, help="Path to aggregated CSV produced by run_many.py")
    parser.add_argument("--outdir", default="figs", help="Directory to save figures")
    parser.add_argument(
        "--metrics",
        default="ALL",
        help="Comma-separated list of metrics to plot, or ALL to include every mean_* column.",
    )
    parser.add_argument(
        "--with_std",
        action="store_true",
        help="Retained for backward compatibility (standard deviations are shown when available).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    metric_arg = args.metrics.strip()
    if metric_arg.upper() == "ALL":
        metric_arg = "all"

    forwarded = [
        "--csv_agg",
        args.csv,
        "--mode",
        "grid",
        "--outdir",
        args.outdir,
        "--metric",
        metric_arg,
    ]

    plot_results.main(forwarded)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
