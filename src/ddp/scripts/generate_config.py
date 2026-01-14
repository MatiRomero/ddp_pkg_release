"""Generate configuration CSV files for grid submissions.

This script materialises a Cartesian product of parameter choices into a
configuration CSV. Each row represents one invocation of
``ddp.scripts.run`` driven through ``run_from_config``.

The default average-duals city directory is ``data/average_duals_city``
relative to the repository root. Override this location by setting the
``DDP_AD_CITY_DIR`` environment variable to an absolute path:

    DDP_AD_CITY_DIR=/path/to/custom/dir python -m ddp.scripts.generate_config
"""

from __future__ import annotations

import csv
import itertools
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Core sweep dimensions -----------------------------------------------------

_DAYS: Sequence[int] = [0,1,2,3,4,5,6,7]
_DEADLINES: Sequence[int] = [10,30,60,90,120,150,180]
_SHADOWS: Sequence[str] = ["naive","pb","hd","ad"]


# Repository root detection --------------------------------------------------

def repo_root() -> Path:
    """Find the repository root by walking up until pyproject.toml is found.
    
    Returns
    -------
    Path
        The repository root directory (parent of pyproject.toml).
        
    Raises
    ------
    RuntimeError
        If pyproject.toml cannot be found by walking up from the current file.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find pyproject.toml by walking up from script location")


def _default_ad_city_dir() -> str:
    """Return the default average-duals city directory path.
    
    The default location is ``data/average_duals_city`` relative to the
    repository root. This can be overridden by setting the ``DDP_AD_CITY_DIR``
    environment variable to an absolute path.
    
    Returns
    -------
    str
        Path to the average-duals city directory (absolute if env var is set,
        otherwise repo-relative).
    """
    env_override = os.environ.get("DDP_AD_CITY_DIR")
    if env_override:
        return str(Path(env_override).expanduser().resolve())
    return str(repo_root() / "data" / "average_duals_city")


# Optional arguments forwarded to ddp.scripts.run --------------------------

# Each entry defines the set of values that should be enumerated for the
# corresponding command-line flag. Leaving the list as [""], [None], or []
# will create an empty column so that the default provided by the run module
# is used.
_OPTIONAL_SWEEP: Mapping[str, Sequence[object]] = {
    "dispatch": ["greedyx"],
    "gamma": [""],
    "tau": [""],
    "plus_gamma": [""],
    "plus_tau": [""],
    "seed": [""],
    "timestamp_column": [""],
    "export_npz": [""],
    "with_opt": [""],
    "opt_method": [""],
    "print_matches": [""],
    "return_details": [""],
    "tie_breaker": [""],
    "reward_type": [""],
    "ad_duals": [_default_ad_city_dir()],
    # "ad_duals": [""],
    "ad_resolution": ["8"],
    # "ad_resolutions": [""],
    "ad_mapping": [""],
    "save_job_csv": [""],
}


def _prompt_for_name() -> str:
    """Request a configuration label from the user.

    Returns
    -------
    str
        The label used to name the output CSV and the companion results
        directory. If the user provides an empty response a timestamp is used
        to guarantee uniqueness.
    """

    response = input("Config name (leave blank to use timestamp): ").strip()
    if response:
        return response
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalise(value: object) -> str:
    """Coerce ``value`` to a trimmed string suitable for CSV emission."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


_SANITISE_PATTERN = re.compile(r"[^A-Za-z0-9]+")


def _sanitise_for_filename(value: str) -> str:
    """Return a filesystem-friendly token derived from ``value``."""

    cleaned = _SANITISE_PATTERN.sub("_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "default"


def _optional_sweep_items() -> Sequence[tuple[str, Sequence[str]]]:
    """Normalise ``_OPTIONAL_SWEEP`` into (column, values) pairs."""

    items: list[tuple[str, Sequence[str]]] = []
    for column, values in _OPTIONAL_SWEEP.items():
        normalised = [_normalise(value) for value in (values or [""])]
        if not normalised:
            normalised = [""]
        items.append((column, normalised))
    return items


def _default_job_csv_path(results_dir: Path, save_name: str) -> Path:
    """Return the default job-level CSV destination for a config row."""

    save_stem = Path(save_name).stem
    return results_dir / f"{save_stem}_jobs.csv"


def _iter_rows(results_dir: Path) -> Iterable[dict[str, str]]:
    """Yield each configuration row as a mapping."""

    optional_items = _optional_sweep_items()
    optional_columns = [column for column, _ in optional_items]

    for day, deadline, shadow in itertools.product(_DAYS, _DEADLINES, _SHADOWS):
        jobs_csv = f"data/meituan_city_lunchtime_plat10301330_day{day}.csv"
        # jobs_csv = f"data/meituan_area6_lunchtime_plat10301330_day{day}.csv"
        base_name_parts = [f"day{day}", f"d{deadline}", shadow]

        option_values_product = itertools.product(
            *(values for _, values in optional_items)
        )
        for option_values in option_values_product:
            if len(optional_columns) != len(option_values):
                raise RuntimeError(
                    "Mismatch between optional column names and values while building"
                    " configuration rows."
                )

            optional_payload = dict(zip(optional_columns, option_values))

            name_parts = list(base_name_parts)
            for column, value in optional_payload.items():
                if value:
                    token = _sanitise_for_filename(value)
                    name_parts.append(f"{column}{token}")

            save_name = "_".join(name_parts) + ".csv"
            save_csv = results_dir / save_name

            default_job_csv = _default_job_csv_path(results_dir, save_name)

            row: dict[str, str] = {
                "day": str(day),
                "d": str(deadline),
                "shadow": shadow,
                "jobs_csv": jobs_csv,
                "save_csv": str(save_csv),
            }
            row.update(optional_payload)
            if not row.get("save_job_csv"):
                row["save_job_csv"] = str(default_job_csv)
            yield row


def main() -> None:
    """Entry point for config generation."""

    name = _prompt_for_name()
    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results") / name
    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = configs_dir / f"config_{name}.csv"

    optional_columns = [column for column, _ in _optional_sweep_items()]
    headers = [
        "day",
        "d",
        "shadow",
        "jobs_csv",
        "save_csv",
        "save_job_csv",
        *[column for column in optional_columns if column != "save_job_csv"],
    ]

    rows = list(_iter_rows(results_dir))

    with config_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {config_path} with {len(rows)} rows.")


if __name__ == "__main__":
    main()
