import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import ddp.scripts.run as run_mod


def _stub_lp(jobs, *_args, **_kwargs):
    return {
        "total_upper": 100.0,
        "duals": np.zeros(len(jobs), dtype=float),
        "method": "stub",
    }


def _stub_simulate(jobs, score_fn, *_args, **_kwargs):
    shadow_vec = np.zeros(len(jobs), dtype=float)
    if score_fn is not None and score_fn.__closure__:
        for cell in score_fn.__closure__:
            contents = cell.cell_contents
            if isinstance(contents, np.ndarray):
                shadow_vec = contents
                break
    savings = float(np.sum(shadow_vec))
    return {
        "total_savings": savings,
        "pooled_pct": 0.0,
        "pairs": [],
        "solos": list(range(len(jobs))),
    }


def _write_job_csv(path: Path) -> None:
    header = [
        "sender_lat",
        "sender_lng",
        "recipient_lat",
        "recipient_lng",
        "platform_order_time",
    ]
    rows = [
        [0.0, 0.0, 1.0, 1.0, "2023-01-01T00:00:00Z"],
        [0.5, 0.5, 1.5, 1.5, "2023-01-01T00:00:01Z"],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_ad_csv(path: Path, values: list[float]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["job_index", "mean_dual"])
        for idx, value in enumerate(values):
            writer.writerow([idx, value])


def test_cli_auto_resolution_selects_best(tmp_path, monkeypatch, capsys):
    jobs_csv = tmp_path / "jobs.csv"
    _write_job_csv(jobs_csv)

    res05 = tmp_path / "example_res05_full.csv"
    res10 = tmp_path / "example_res10_full.csv"
    _write_ad_csv(res05, [1.0, 1.0])
    _write_ad_csv(res10, [3.0, 3.0])

    before_mtimes = {path: path.stat().st_mtime_ns for path in (res05, res10)}

    monkeypatch.setattr(run_mod, "compute_lp_relaxation", _stub_lp)
    monkeypatch.setattr(run_mod, "simulate", _stub_simulate)

    output_csv = tmp_path / "results.csv"

    argv = [
        "python",
        "--jobs-csv",
        str(jobs_csv),
        "--timestamp-column",
        "platform_order_time",
        "--d",
        "60",
        "--seed",
        "0",
        "--shadows",
        "ad",
        "--dispatch",
        "greedy",
        "--ad_duals",
        str(res05),
        "--save_csv",
        str(output_csv),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_mod.main()

    captured = capsys.readouterr()
    assert "[ad] ad resolution 05" in captured.out
    assert "[ad] ad resolution 10" in captured.out

    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "expected CSV output"
    ad_resolutions = {row["dispatch"]: row["ad_resolution"] for row in rows}
    assert ad_resolutions == {"greedy": "10"}
    savings_values = {row["dispatch"]: float(row["savings"]) for row in rows}
    assert savings_values == {"greedy": 6.0}

    after_mtimes = {path: path.stat().st_mtime_ns for path in (res05, res10)}
    assert before_mtimes == after_mtimes
