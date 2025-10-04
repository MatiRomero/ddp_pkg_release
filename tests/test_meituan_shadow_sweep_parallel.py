from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(module: str, csv_path: Path, out_csv: Path) -> None:
    command = [
        sys.executable,
        "-m",
        module,
        "--jobs-csv",
        str(csv_path),
        "--limit",
        "3",
        "--gamma-values",
        "1",
        "--tau-values",
        "0",
        "--shadows",
        "naive",
        "--dispatch",
        "greedy",
        "--no-progress",
        "--csv",
        str(out_csv),
    ]
    if module.endswith("_parallel"):
        command.extend(["--workers", "1"])
    env = os.environ.copy()
    src_path = str(_repo_root() / "src")
    env["PYTHONPATH"] = (
        f"{src_path}:{env['PYTHONPATH']}" if "PYTHONPATH" in env else src_path
    )
    subprocess.run(command, check=True, cwd=_repo_root(), env=env)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    rows.sort(key=lambda row: (row["geometry"], row["shadow"], row["gamma"], row["tau"]))
    return rows


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_parallel_cli_matches_sequential(tmp_path: Path) -> None:
    csv_path = _repo_root() / "tests" / "data" / "sample_jobs.csv"
    seq_csv = tmp_path / "sequential.csv"
    par_csv = tmp_path / "parallel.csv"

    _run_cli("ddp.scripts.meituan_shadow_sweep", csv_path, seq_csv)
    _run_cli("ddp.scripts.meituan_shadow_sweep_parallel", csv_path, par_csv)

    assert seq_csv.exists()
    assert par_csv.exists()

    sequential_rows = _load_csv(seq_csv)
    parallel_rows = _load_csv(par_csv)

    assert sequential_rows == parallel_rows

