"""Co-ordinate Meituan *shadow* sweeps via an intermediate configuration CSV.

This helper keeps the legacy ``run_meituan_shadow_sweep`` batch script
untouched while providing a higher-level orchestrator that mirrors its
interface, materialises the run matrix as a CSV (``generate_config`` style),
executes each configuration sequentially (``run_from_config`` style) and
finally collapses the per-run CSV files into a single ``final_result`` table.

Example usage::

    python scripts/run_meituan_shadow_sweep_config.py --day 0 --day 1 --d 20 \
        --gamma-values 0.8:1.2:0.2 --tau-values 0,5e-2 --with_opt

Any unrecognised flags are forwarded to ``ddp.scripts.meituan_shadow_sweep`` so callers can
enable options such as ``--with_opt`` or ``--export-npz`` without this helper
having to mirror the entire CLI surface.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ddp.results.append_csv import append_csv_locked


LOGGER = logging.getLogger(__name__)


DEFAULT_DAYS = list(range(8))
DEFAULT_D_VALUES = [10, 20, 30]
DEFAULT_JOBS_TEMPLATE = "data/meituan_city_lunchtime_plat10301330_day{day}.csv"
DEFAULT_RESULTS_DIR = Path("results/meituan_shadow_runs")
DEFAULT_CONFIG_PATH = Path("configs/meituan_shadow_sweep_config.csv")
DEFAULT_FINAL_OUT = Path("results/meituan_shadow_final_result.csv")


@dataclass(frozen=True)
class ConfigRow:
    """Single row in the generated configuration CSV."""

    day: int
    d: float
    jobs_csv: Path
    save_csv: Path
    shadows: str
    gamma: float
    tau: float

    def to_csv_row(self) -> dict[str, str]:
        """Return a mapping ready for :class:`csv.DictWriter`."""

        return {
            "day": str(self.day),
            "d": str(self.d),
            "shadows": self.shadows,
            "jobs_csv": str(self.jobs_csv),
            "save_csv": str(self.save_csv),
            "gamma": repr(self.gamma),
            "tau": repr(self.tau),
        }


def _parse_values(spec: str) -> list[float]:
    """Parse ``1,2,3`` or ``start:stop:step`` into a list of floats."""

    cleaned = spec.strip()
    if not cleaned:
        raise ValueError("empty specification")
    if ":" in cleaned:
        start_s, stop_s, *rest = cleaned.split(":")
        step_s = rest[0] if rest else "1"
        start = float(start_s)
        stop = float(stop_s)
        step = float(step_s)
        if step == 0:
            raise ValueError("step must be non-zero")
        count = int(round((stop - start) / step)) + 1
        return [start + idx * step for idx in range(count)]
    return [float(chunk) for chunk in cleaned.split(",") if chunk.strip()]


def _format_token(value: float) -> str:
    """Create a filesystem-friendly token for ``value``."""

    text = f"{value:.12g}"  # trim scientific notation where possible
    text = text.replace("-", "m")
    text = text.replace(".", "p")
    return text


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for ``path`` if it doesn't exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--day",
        type=int,
        action="append",
        dest="days",
        help="Specific day values to run (repeatable). Defaults to 0..7.",
    )
    parser.add_argument(
        "--d",
        type=float,
        action="append",
        dest="d_values",
        help="Specific d values to run (repeatable). Defaults to 10,20,30.",
    )
    parser.add_argument(
        "--jobs-template",
        default=DEFAULT_JOBS_TEMPLATE,
        help=(
            "Template for the jobs CSV path. The token '{day}' is replaced with"
            " the day index."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where individual run CSVs will be stored.",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Destination for the generated configuration CSV.",
    )
    parser.add_argument(
        "--final-out",
        type=Path,
        default=DEFAULT_FINAL_OUT,
        help="Location for the aggregated final_result CSV.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="platform_order_time",
        help="Timestamp column passed to ddp.scripts.meituan_shadow_sweep.",
    )
    parser.add_argument(
        "--gamma-values",
        default="1.0",
        help="Comma list or start:stop:step grid for gamma values.",
    )
    parser.add_argument(
        "--tau-values",
        default="0.0",
        help="Comma list or start:stop:step grid for tau values.",
    )
    parser.add_argument(
        "--shadows",
        default="hd",
        help=(
            "Shadow families forwarded to ddp.scripts.meituan_shadow_sweep"
            " (default: hd)."
        ),
    )
    parser.add_argument(
        "--keep-individual",
        action="store_true",
        help="Keep individual CSV results after building the final aggregate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the config CSV and print planned commands without executing.",
    )
    return parser


def _resolve_days(values: Sequence[int] | None) -> list[int]:
    return list(values) if values else list(DEFAULT_DAYS)


def _resolve_d_values(values: Sequence[float] | None) -> list[float]:
    return list(values) if values else list(DEFAULT_D_VALUES)


def _build_config_rows(
    *,
    days: Iterable[int],
    d_values: Iterable[float],
    jobs_template: str,
    results_dir: Path,
    shadows: str,
    gamma_grid: Sequence[float],
    tau_grid: Sequence[float],
) -> list[ConfigRow]:
    rows: list[ConfigRow] = []
    for day in days:
        jobs_path = Path(jobs_template.format(day=day)).resolve()
        if not jobs_path.exists():
            LOGGER.warning("Skipping day %s because jobs CSV is missing: %s", day, jobs_path)
            continue
        for d_value in d_values:
            for gamma in gamma_grid:
                for tau in tau_grid:
                    save_path = results_dir / (
                        "_".join(
                            [
                                f"meituan_day{day}",
                                f"d{_format_token(d_value)}",
                                f"gamma{_format_token(gamma)}",
                                f"tau{_format_token(tau)}",
                            ]
                        )
                        + ".csv"
                    )
                    rows.append(
                        ConfigRow(
                            day=day,
                            d=d_value,
                            jobs_csv=jobs_path,
                            save_csv=save_path,
                            shadows=shadows,
                            gamma=gamma,
                            tau=tau,
                        )
                    )
    return rows


def _write_config_csv(config_path: Path, rows: list[ConfigRow]) -> None:
    _ensure_parent(config_path)
    fieldnames = ["day", "d", "shadows", "jobs_csv", "save_csv", "gamma", "tau"]

    with config_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _load_config_rows(config_path: Path) -> list[dict[str, str]]:
    with config_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _run_configuration(
    *,
    row: dict[str, str],
    forward_args: Sequence[str],
    timestamp_column: str,
    dry_run: bool,
) -> subprocess.CompletedProcess[bytes]:
    save_csv = Path(row["save_csv"])
    _ensure_parent(save_csv)
    jobs_csv = row["jobs_csv"]
    shadows = row.get("shadows", "hd") or "hd"
    cmd = [
        sys.executable,
        "-m",
        "ddp.scripts.meituan_shadow_sweep",
        *forward_args,
        "--jobs-csv",
        jobs_csv,
        "--timestamp-column",
        timestamp_column,
        "--d",
        row["d"],
        "--csv",
        str(save_csv),
    ]

    if shadows:
        cmd.extend(["--shadows", shadows])

    gamma = row.get("gamma", "").strip()
    tau = row.get("tau", "").strip()
    if gamma:
        cmd.extend(["--gamma-values", gamma])
    if tau:
        cmd.extend(["--tau-values", tau])

    LOGGER.info("Running configuration: %s", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0)

    return subprocess.run(cmd, check=False)


def _aggregate_results(
    *,
    csv_paths: Iterable[Path],
    final_out: Path,
    keep_individual: bool,
) -> None:
    if final_out.exists():
        final_out.unlink()

    _ensure_parent(final_out)
    for path in csv_paths:
        if not path.exists():
            LOGGER.warning("Skipping missing CSV %s during aggregation", path)
            continue
        append_csv_locked(str(path), str(final_out))
        if not keep_individual:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if not final_out.exists():
        LOGGER.warning("No CSV files were aggregated into %s", final_out)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args, forward = parser.parse_known_args(argv)

    days = _resolve_days(args.days)
    d_values = _resolve_d_values(args.d_values)

    try:
        gamma_grid = _parse_values(args.gamma_values)
    except ValueError as exc:
        parser.error(f"invalid --gamma-values: {exc}")
    try:
        tau_grid = _parse_values(args.tau_values)
    except ValueError as exc:
        parser.error(f"invalid --tau-values: {exc}")

    config_rows = _build_config_rows(
        days=days,
        d_values=d_values,
        jobs_template=args.jobs_template,
        results_dir=args.results_dir,
        shadows=args.shadows,
        gamma_grid=gamma_grid,
        tau_grid=tau_grid,
    )

    if not config_rows:
        LOGGER.warning("No configurations produced; nothing to do")
        return

    _write_config_csv(args.config_out, config_rows)
    LOGGER.info("Wrote configuration CSV with %s rows to %s", len(config_rows), args.config_out)

    rows = _load_config_rows(args.config_out)

    produced_csvs: list[Path] = []
    for index, row in enumerate(rows, start=1):
        LOGGER.info("[%s/%s] Processing configuration", index, len(rows))
        result = _run_configuration(
            row=row,
            forward_args=forward,
            timestamp_column=args.timestamp_column,
            dry_run=args.dry_run,
        )
        if result.returncode != 0:
            LOGGER.error("Run failed for row %s with return code %s", index, result.returncode)
            raise SystemExit(result.returncode)
        produced_csvs.append(Path(row["save_csv"]))

    if args.dry_run:
        LOGGER.info("Dry run complete; skipping aggregation")
        return

    _aggregate_results(
        csv_paths=produced_csvs,
        final_out=args.final_out,
        keep_individual=args.keep_individual,
    )
    LOGGER.info("Aggregated %s CSV files into %s", len(produced_csvs), args.final_out)


if __name__ == "__main__":
    main()

