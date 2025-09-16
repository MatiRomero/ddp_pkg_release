#!/usr/bin/env python3
"""Compatibility wrapper forwarding to :mod:`ddp.scripts.plot_results`."""

from __future__ import annotations

import argparse
from typing import Sequence

from ddp.scripts import plot_results


METRICS_ALL = [
    "mean_savings",
    "mean_pooled_pct",
    "mean_ratio_lp",
    "mean_ratio_opt",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot metric(s) vs parameter value from aggregated sweep CSV (legacy wrapper)."
    )
    parser.add_argument("--csv", required=True, help="Path to aggregated CSV results")
    parser.add_argument(
        "--metric",
        default="all",
        choices=["all"] + METRICS_ALL,
        help="Metric to plot (all → one figure per metric).",
    )
    parser.add_argument(
        "--include_policies",
        default="",
        help='Comma-separated list like "shadow+dispatch" to restrict policies.',
    )
    parser.add_argument("--title", default="", help="Optional custom title for single-metric plots")
    parser.add_argument(
        "--out",
        default="",
        help="Output path or directory. If omitted, figures are written to figs/<csv>_<metric>.png",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    forwarded = [
        "--csv_agg",
        args.csv,
        "--mode",
        "sweep",
        "--metric",
        args.metric,
    ]

    if args.include_policies:
        forwarded.extend(["--include_policies", args.include_policies])
    if args.title:
        forwarded.extend(["--title", args.title])
    if args.out:
        forwarded.extend(["--out", args.out])

    plot_results.main(forwarded)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
