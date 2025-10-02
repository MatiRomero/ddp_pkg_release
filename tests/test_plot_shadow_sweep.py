from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ddp.scripts import plot_shadow_sweep


def test_best_shadow_summary_written(tmp_path, capsys):
    csv_path = tmp_path / "sweep.csv"
    data = pd.DataFrame(
        [
            {
                "geometry": "plane",
                "d": 300,
                "resolution": 7,
                "dispatch": "rbatch",
                "shadow": "ad",
                "gamma": 0.1,
                "tau": 0.5,
                "metric": "savings",
                "mean": 10.0,
            },
            {
                "geometry": "plane",
                "d": 300,
                "resolution": 7,
                "dispatch": "rbatch",
                "shadow": "ad",
                "gamma": 0.2,
                "tau": 0.5,
                "metric": "savings",
                "mean": 12.0,
            },
            {
                "geometry": "plane",
                "d": 300,
                "resolution": 7,
                "dispatch": "rbatch",
                "shadow": "pb",
                "gamma": 0.1,
                "tau": 0.5,
                "metric": "savings",
                "mean": 8.0,
            },
            {
                "geometry": "plane",
                "d": 300,
                "resolution": 7,
                "dispatch": "rbatch",
                "shadow": "pb",
                "gamma": 0.2,
                "tau": 0.5,
                "metric": "savings",
                "mean": 9.0,
            },
        ]
    )
    data.to_csv(csv_path, index=False)

    best_out = tmp_path / "best.csv"
    plot_shadow_sweep.main(
        [
            "--csv",
            str(csv_path),
            "--metric",
            "savings",
            "--best-out",
            str(best_out),
        ]
    )

    best_df = pd.read_csv(best_out)
    expected = pd.DataFrame(
        [
            {
                "geometry": "plane",
                "d": 300.0,
                "resolution": 7.0,
                "dispatch": "rbatch",
                "shadow": "ad",
                "best_gamma": 0.2,
                "best_tau": 0.5,
                "best_mean": 12.0,
            },
            {
                "geometry": "plane",
                "d": 300.0,
                "resolution": 7.0,
                "dispatch": "rbatch",
                "shadow": "pb",
                "best_gamma": 0.2,
                "best_tau": 0.5,
                "best_mean": 9.0,
            },
        ]
    )

    pd.testing.assert_frame_equal(
        best_df.sort_values(["shadow"]).reset_index(drop=True),
        expected.sort_values(["shadow"]).reset_index(drop=True),
        check_dtype=False,
    )

    captured = capsys.readouterr()
    assert "Best for geometry=plane, d=300, resolution=7, dispatch=rbatch, shadow=ad" in captured.out
    assert str(best_out) in captured.out
