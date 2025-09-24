"""Inspect aggregated average-dual coverage and statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Hashable, Iterable

import numpy as np

from ddp.scripts.average_duals import _load_mapping
from ddp.scripts.build_average_duals import _Stats, compute_average_duals


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hd_csv",
        nargs="?",
        type=Path,
        default=Path("data/hd_dataset_n100_d10.csv"),
        help="Hindsight-dual dataset to analyse (default: data/hd_dataset_n100_d10.csv)",
    )
    parser.add_argument(
        "--mapping",
        default="ddp.mappings.uniform_grid:mapping",
        help="module:callable specification for the coordinate-based type mapping",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        help="Optional path to save an origin-cell coverage heatmap (requires matplotlib)",
    )
    return parser


def _summarise_stats(stats: dict[Hashable, _Stats]) -> list[tuple[str, int, float, float]]:
    rows: list[tuple[str, int, float, float]] = []
    for key in sorted(stats.keys(), key=repr):
        bucket = stats[key]
        rows.append((str(key), bucket.count, bucket.mean, bucket.std_dev))
    return rows


def _format_table(rows: Iterable[tuple[str, int, float, float]]) -> str:
    rows = list(rows)
    if not rows:
        return "No buckets observed."

    type_width = max(len("type"), *(len(row[0]) for row in rows))
    count_width = max(len("count"), *(len(str(row[1])) for row in rows))

    header = f"{'type':<{type_width}}  {'count':>{count_width}}  mean_dual    std_dev"
    lines = [header, "-" * len(header)]
    for key, count, mean, std_dev in rows:
        lines.append(
            f"{key:<{type_width}}  {count:>{count_width}d}  {mean:10.6f}  {std_dev:10.6f}"
        )
    return "\n".join(lines)


def _make_origin_heatmap(
    stats: dict[Hashable, _Stats]
) -> tuple[np.ndarray, list[int], list[int]]:
    origin_counts: dict[tuple[int, int], int] = {}
    for key, bucket in stats.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        origin = key[0]
        if not isinstance(origin, tuple) or len(origin) != 2:
            continue
        origin_counts[origin] = origin_counts.get(origin, 0) + bucket.count

    if not origin_counts:
        raise ValueError("Heatmap requires tuple-based uniform-grid keys")

    x_indices = sorted({origin[0] for origin in origin_counts})
    y_indices = sorted({origin[1] for origin in origin_counts})
    x_lookup = {value: idx for idx, value in enumerate(x_indices)}
    y_lookup = {value: idx for idx, value in enumerate(y_indices)}

    heatmap = np.zeros((len(y_indices), len(x_indices)), dtype=float)
    for (ix, iy), count in origin_counts.items():
        heatmap[y_lookup[iy], x_lookup[ix]] = float(count)
    return heatmap, x_indices, y_indices


def _save_heatmap(path: Path, heatmap: np.ndarray, x_ticks: list[int], y_ticks: list[int]) -> None:
    import matplotlib.pyplot as plt  # Imported lazily so the CLI works without plotting extras

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, origin="lower", cmap="viridis")
    ax.set_xticks(range(len(x_ticks)), labels=[str(val) for val in x_ticks])
    ax.set_yticks(range(len(y_ticks)), labels=[str(val) for val in y_ticks])
    ax.set_xlabel("origin_x grid index")
    ax.set_ylabel("origin_y grid index")
    ax.set_title("Uniform grid origin coverage")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Job count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    mapping, _expected = _load_mapping(args.mapping)
    stats = compute_average_duals(args.hd_csv, mapping)
    rows = _summarise_stats(stats)
    print(_format_table(rows))

    if args.heatmap:
        heatmap, x_ticks, y_ticks = _make_origin_heatmap(stats)
        _save_heatmap(args.heatmap, heatmap, x_ticks, y_ticks)
        print(f"Saved heatmap to {args.heatmap}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
