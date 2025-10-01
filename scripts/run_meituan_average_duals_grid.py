"""Batch runner for Meituan average-dual generation."""

from __future__ import annotations

import argparse
import itertools
import logging
from decimal import Decimal
from pathlib import Path
from typing import Callable, Sequence, TypeVar

from ddp.scripts.meituan_average_duals import (
    _parse_day_list,
    _format_deadline,
    build_average_duals,
    save_summary_map,
)

DEFAULT_EXPORT_BASE = Path("data/average_duals")
DEFAULT_EXPORT_STEM = "meituan_ad_day{day}_d{d}_res{r}"
DEFAULT_EXPORT_SUMMARY_TEMPLATE = str(
    DEFAULT_EXPORT_BASE / f"{DEFAULT_EXPORT_STEM}_summary.csv"
)
DEFAULT_EXPORT_TARGET_TEMPLATE = str(
    DEFAULT_EXPORT_BASE / f"{DEFAULT_EXPORT_STEM}_full.csv"
)

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


def _parse_numeric_list(
    value: str,
    *,
    cast: Callable[[str], T],
    default_step: Decimal,
) -> list[T]:
    """Parse comma-separated values or inclusive ranges into a list."""

    def _cast_decimal(text: str) -> Decimal:
        try:
            return Decimal(text)
        except Exception as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"Invalid numeric literal: {text!r}") from exc

    result: list[T] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue

        # Support "start-end[:step]" range syntax.
        range_and_step = item.split(":", 1)
        range_part = range_and_step[0]
        step = (
            _cast_decimal(range_and_step[1])
            if len(range_and_step) == 2
            else default_step
        )

        if "-" in range_part and not range_part.startswith("-"):
            start_text, end_text = range_part.split("-", 1)
            start = _cast_decimal(start_text)
            end = _cast_decimal(end_text)
            if step == 0:
                raise argparse.ArgumentTypeError("Range step must be non-zero")

            values: list[Decimal] = []
            current = start
            comparator: Callable[[Decimal, Decimal], bool]
            if step > 0:
                comparator = lambda a, b: a <= b
            else:
                comparator = lambda a, b: a >= b

            # Guard against infinite loops; 10_000 iterations should be ample.
            for _ in range(10_000):
                if not comparator(current, end):
                    break
                values.append(current)
                current += step
            else:  # pragma: no cover - defensive
                raise argparse.ArgumentTypeError(
                    f"Range '{item}' produced too many values; check the step"
                )

            if not values:
                raise argparse.ArgumentTypeError(
                    f"Range '{item}' did not produce any values"
                )

            for value_decimal in values:
                result.append(cast(str(value_decimal)))
            continue

        result.append(cast(item))

    if not result:
        raise argparse.ArgumentTypeError("Expected at least one value")
    return result


def _parse_day_grid(value: str) -> list[int]:
    return _parse_numeric_list(value, cast=lambda x: int(float(x)), default_step=Decimal(1))


def _parse_deadline_grid(value: str) -> list[float]:
    def _cast(text: str) -> float:
        try:
            return float(text)
        except ValueError as exc:  # pragma: no cover - defensive
            raise argparse.ArgumentTypeError(f"Invalid deadline value: {text!r}") from exc

    return _parse_numeric_list(value, cast=_cast, default_step=Decimal(1))


def _parse_resolution_grid(value: str) -> list[int]:
    return _parse_numeric_list(value, cast=lambda x: int(float(x)), default_step=Decimal(1))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--days",
        required=True,
        help=(
            "Comma-separated list or inclusive ranges of target days (e.g. '1,2,5-7')."
        ),
        type=_parse_day_grid,
    )
    parser.add_argument(
        "--deadlines",
        required=True,
        help=(
            "Comma-separated list or inclusive ranges of deadline values "
            "(e.g. '15,20,25' or '10-30:5')."
        ),
        type=_parse_deadline_grid,
    )
    parser.add_argument(
        "--resolutions",
        required=True,
        help=(
            "Comma-separated list or inclusive ranges of H3 resolutions (e.g. '7-9')."
        ),
        type=_parse_resolution_grid,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing Meituan CSV snapshots",
    )
    parser.add_argument(
        "--jobs-pattern",
        default="meituan_city_lunchtime_plat10301330_day{day}.csv",
        help="Filename pattern with '{day}' placeholder for snapshots",
    )
    parser.add_argument(
        "--timestamp-column",
        default="platform_order_time",
        help="Timestamp column name in the CSV files",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=0,
        help="Radius for neighbour search when a hex pair is missing",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["hd", "zero", "nan"],
        default="hd",
        help="Fallback when a hex pair has no historical average",
    )
    parser.add_argument(
        "--history-days",
        type=_parse_day_list,
        help="Comma-separated list of history days (default: all except target)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/hd_cache"),
        help="Directory for cached per-day HD dual CSV files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute HD duals even when cached files exist",
    )
    parser.add_argument(
        "--export-summary",
        type=str,
        help=(
            "Optional template for CSV path of aggregated average-dual tables "
            f"(default: {DEFAULT_EXPORT_SUMMARY_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--export-target",
        type=str,
        help=(
            "Optional template for CSV path of target day with attached AD means "
            f"(default: {DEFAULT_EXPORT_TARGET_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--folium-map",
        type=str,
        help="Optional template for HTML path of a Folium sender coverage map",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info-level logging",
    )
    return parser


def _format_template(path_template: str | None, context: dict[str, object]) -> Path | None:
    if not path_template:
        return None
    try:
        rendered = path_template.format(**context)
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown template placeholder in {path_template!r}: {exc}") from exc
    path = Path(rendered)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _log_combination(day: int, deadline: float, resolution: int) -> str:
    return (
        f"day={day} deadline={_format_deadline(deadline)} resolution={resolution}"
    )


def run_grid(
    *,
    days: Sequence[int],
    deadlines: Sequence[float],
    resolutions: Sequence[int],
    data_dir: Path,
    jobs_pattern: str,
    timestamp_column: str,
    neighbor_radius: int,
    missing_policy: str,
    history_days: Sequence[int] | None,
    cache_dir: Path,
    force: bool,
    export_summary_template: str | None,
    export_target_template: str | None,
    folium_map_template: str | None,
) -> None:
    total = len(days) * len(deadlines) * len(resolutions)
    LOGGER.info("Running %s parameter combinations", total)

    for index, (day, deadline, resolution) in enumerate(
        itertools.product(days, deadlines, resolutions), start=1
    ):
        context = {
            "day": day,
            "deadline": deadline,
            "d": _format_deadline(deadline),
            "resolution": resolution,
            "r": resolution,
        }

        LOGGER.info(
            "[%s/%s] Building averages for %s", index, total, _log_combination(day, deadline, resolution)
        )

        result = build_average_duals(
            day=int(day),
            data_dir=data_dir,
            jobs_pattern=jobs_pattern,
            cache_dir=cache_dir,
            timestamp_column=timestamp_column,
            deadline=float(deadline),
            resolution=int(resolution),
            history_days=list(history_days) if history_days is not None else None,
            neighbor_radius=int(neighbor_radius),
            missing_policy=missing_policy,
            force=bool(force),
        )

        summary_path = _format_template(export_summary_template, context)
        if summary_path is not None:
            result.summary.to_csv(summary_path, index=False)
            LOGGER.info("Wrote summary to %s", summary_path)

        target_path = _format_template(export_target_template, context)
        if target_path is not None:
            result.target.to_csv(target_path, index=False)
            LOGGER.info("Wrote target to %s", target_path)

        folium_path = _format_template(folium_map_template, context)
        if folium_path is not None:
            save_summary_map(result.summary, folium_path)
            LOGGER.info("Wrote Folium map to %s", folium_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    export_summary_template = args.export_summary or DEFAULT_EXPORT_SUMMARY_TEMPLATE
    export_target_template = args.export_target or DEFAULT_EXPORT_TARGET_TEMPLATE

    run_grid(
        days=args.days,
        deadlines=args.deadlines,
        resolutions=args.resolutions,
        data_dir=args.data_dir,
        jobs_pattern=str(args.jobs_pattern),
        timestamp_column=str(args.timestamp_column),
        neighbor_radius=int(args.neighbor_radius),
        missing_policy=str(args.missing_policy),
        history_days=args.history_days,
        cache_dir=args.cache_dir,
        force=bool(args.force),
        export_summary_template=export_summary_template,
        export_target_template=export_target_template,
        folium_map_template=args.folium_map,
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
