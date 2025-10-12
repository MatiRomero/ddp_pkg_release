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

from ddp.scripts import plot_results


def test_format_policy_label_opt_opt_collapses_to_single_label():
    assert plot_results._format_policy_label("opt+opt") == "OPT"


def test_format_policy_label_known_shadow_and_dispatch_are_formatted():
    formatted = plot_results._format_policy_label("naive+rbatch")
    assert formatted == "NAIVE + RBAT"


def test_format_policy_label_falls_back_to_uppercase_for_unknown_values():
    formatted = plot_results._format_policy_label("pb-fast+lookahead")
    assert formatted == "PB-FAST + LOOKAHEAD"


def test_time_metric_sweep_uses_log_scales(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    original_subplots = plot_results.plt.subplots

    def wrapped_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plot_results.plt, "subplots", wrapped_subplots)

    df = pd.DataFrame(
        [
            {
                "param": "d",
                "param_value": 10.0,
                "shadow": "opt",
                "dispatch": "opt",
                "mean_time-s": 5.0,
                "trial_count": 5,
            },
            {
                "param": "d",
                "param_value": 20.0,
                "shadow": "opt",
                "dispatch": "opt",
                "mean_time-s": 10.0,
                "trial_count": 5,
            },
        ]
    )

    out_path = tmp_path / "plot.png"
    plot_results._plot_metric_sweep(
        df,
        "mean_time-s",
        include=None,
        title=None,
        csv_hint=None,
        out_arg=str(out_path),
    )

    ax = captured["ax"]
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
