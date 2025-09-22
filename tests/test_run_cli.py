from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ddp.scripts.csv_loader import load_jobs_from_csv


def _jobs_to_arrays(jobs):
    origins = np.array([job.origin for job in jobs], dtype=float)
    dests = np.array([job.dest for job in jobs], dtype=float)
    timestamps = np.array([job.timestamp for job in jobs], dtype=float)
    return origins, dests, timestamps


def test_cli_csv_and_npz_roundtrip(tmp_path):
    base_args = [
        sys.executable,
        "-m",
        "ddp.scripts.run",
        "--d",
        "300",
        "--seed",
        "0",
        "--shadows",
        "naive",
        "--dispatch",
        "greedy",
    ]

    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing_path}" if existing_path else str(src_dir)
    )

    csv_path = Path(__file__).with_name("data").joinpath("sample_jobs.csv")
    jobs = load_jobs_from_csv(csv_path)
    origins, dests, timestamps = _jobs_to_arrays(jobs)

    base_npz = tmp_path / "input_jobs.npz"
    np.savez(base_npz, origins=origins, dests=dests, timestamps=timestamps)

    csv_export = tmp_path / "csv_roundtrip.npz"
    npz_export = tmp_path / "npz_roundtrip.npz"

    csv_cmd = base_args + [
        "--jobs-csv",
        str(csv_path),
        "--timestamp-column",
        "platform_order_time",
        "--export-npz",
        str(csv_export),
    ]
    result_csv = subprocess.run(
        csv_cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=root_dir,
        env=env,
    )
    assert result_csv.returncode == 0

    npz_cmd = base_args + [
        "--jobs",
        str(base_npz),
        "--export-npz",
        str(npz_export),
    ]
    result_npz = subprocess.run(
        npz_cmd,
        check=True,
        capture_output=True,
        text=True,
        cwd=root_dir,
        env=env,
    )
    assert result_npz.returncode == 0

    with np.load(csv_export) as csv_data, np.load(npz_export) as npz_data:
        for key in ("origins", "dests", "timestamps"):
            np.testing.assert_allclose(csv_data[key], npz_data[key])
            np.testing.assert_allclose(csv_data[key], {
                "origins": origins,
                "dests": dests,
                "timestamps": timestamps,
            }[key])
