import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import run_instance


def _sample_jobs() -> list:
    csv_path = Path(__file__).with_name("data").joinpath("sample_jobs.csv")
    return load_jobs_from_csv(csv_path)


def test_run_instance_writes_job_csv(tmp_path) -> None:
    jobs = _sample_jobs()
    job_csv = tmp_path / "dispatch_rows.csv"

    run_instance(
        jobs,
        d=300,
        shadows=("naive",),
        dispatches=("rbatch",),
        seed=0,
        with_opt=False,
        save_csv="",
        save_job_csv=str(job_csv),
        print_table=False,
    )

    assert job_csv.exists()

    with job_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert rows, "expected job CSV to contain rows"
    assert len(rows) == len(jobs)

    for row in rows:
        assert row["dispatch"] == "rbatch"
        assert row["shadow"] == "naive"
        idx = int(row["job_index"])
        job = jobs[idx]

        assert math.isclose(float(row["origin_lat"]), job.origin[0])
        assert math.isclose(float(row["origin_lng"]), job.origin[1])
        assert math.isclose(float(row["dest_lat"]), job.dest[0])
        assert math.isclose(float(row["dest_lng"]), job.dest[1])

        arrival = float(row["arrival_time"])
        dispatch_time = float(row["dispatch_time"])
        delay = float(row["delay"])
        assert math.isclose(delay, dispatch_time - arrival)
        assert dispatch_time >= arrival

