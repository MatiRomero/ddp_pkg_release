import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ddp.scripts.aggregate_results import aggregate
from ddp.scripts.csv_loader import load_jobs_from_csv
from ddp.scripts.run import run_instance, run_once


def _sample_jobs() -> list:
    csv_path = Path(__file__).with_name("data").joinpath("sample_jobs.csv")
    return load_jobs_from_csv(csv_path)


def test_run_instance_records_effective_gamma(tmp_path) -> None:
    jobs = _sample_jobs()
    out_csv = tmp_path / "rows.csv"

    result = run_instance(
        jobs,
        d=300,
        shadows=("naive",),
        dispatches=("greedy", "batch+"),
        seed=0,
        with_opt=False,
        save_csv=str(out_csv),
        print_table=False,
        gamma=2.5,
        tau=-0.5,
        gamma_plus=1.75,
        tau_plus=0.25,
    )

    rows = {row["dispatch"]: row for row in result["rows"]}
    assert np.isclose(rows["greedy"]["gamma"], 2.5)
    assert np.isclose(rows["greedy"]["tau"], -0.5)
    assert rows["greedy"]["gamma_plus"] is None
    assert rows["greedy"]["tau_plus"] is None
    assert rows["greedy"].get("tau_s") is None

    assert rows["batch+"]["gamma"] is None
    assert rows["batch+"]["tau"] is None
    assert np.isclose(rows["batch+"]["gamma_plus"], 1.75)
    assert np.isclose(rows["batch+"]["tau_plus"], 0.25)
    assert rows["batch+"].get("tau_s") is None

    with out_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        for header in ("gamma", "tau", "gamma_plus", "tau_plus", "tau_s"):
            assert header in reader.fieldnames


def test_run_once_includes_gamma_values() -> None:
    row = run_once(
        n=6,
        d=300,
        dispatch="rbatch+",
        shadow="naive",
        seed=0,
        gamma=0.5,
        tau=0.1,
        gamma_plus=1.2,
        tau_plus=-0.4,
    )

    assert row["dispatch"] == "rbatch+"
    assert row["gamma"] is None
    assert row["tau"] is None
    assert np.isclose(row["gamma_plus"], 1.2)
    assert np.isclose(row["tau_plus"], -0.4)
    assert row.get("tau_s") is None


def test_aggregate_groups_by_gamma(tmp_path) -> None:
    fieldnames = [
        "param",
        "param_value",
        "shadow",
        "dispatch",
        "gamma",
        "tau",
        "gamma_plus",
        "tau_plus",
        "tau_s",
        "n",
        "d",
        "seed",
        "savings",
        "pooled_pct",
        "ratio_lp",
        "lp_gap",
        "ratio_opt",
        "opt_gap",
        "pairs",
        "solos",
        "time_s",
        "method",
    ]
    csv_path = tmp_path / "input.csv"
    rows = [
        {
            "param": "gamma",
            "param_value": 1,
            "shadow": "naive",
            "dispatch": "greedy",
            "gamma": 1.0,
            "tau": 0.0,
            "gamma_plus": "",
            "tau_plus": "",
            "tau_s": "",
            "n": 10,
            "d": 300,
            "seed": 0,
            "savings": 5.0,
            "pooled_pct": 50.0,
            "ratio_lp": 0.5,
            "lp_gap": 1.0,
            "ratio_opt": 0.5,
            "opt_gap": 1.0,
            "pairs": 2,
            "solos": 1,
            "time_s": 0.1,
            "method": "score",
        },
        {
            "param": "gamma",
            "param_value": 1,
            "shadow": "naive",
            "dispatch": "greedy",
            "gamma": 2.0,
            "tau": 0.0,
            "gamma_plus": "",
            "tau_plus": "",
            "tau_s": "",
            "n": 10,
            "d": 300,
            "seed": 1,
            "savings": 6.0,
            "pooled_pct": 55.0,
            "ratio_lp": 0.6,
            "lp_gap": 0.9,
            "ratio_opt": 0.6,
            "opt_gap": 0.9,
            "pairs": 3,
            "solos": 1,
            "time_s": 0.2,
            "method": "score",
        },
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    agg = aggregate(str(csv_path))
    greedy = agg[agg["dispatch"] == "greedy"]
    assert set(greedy["gamma"].dropna()) == {1.0, 2.0}
    assert len(greedy) == 2

