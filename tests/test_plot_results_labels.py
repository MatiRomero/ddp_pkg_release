from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
from matplotlib import colors as mcolors
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


def test_sweep_styles_follow_dispatch_and_shadow_patterns(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    calls: list[dict[str, object]] = []

    original_errorbar = Axes.errorbar

    def capture(self, *args, **kwargs):
        calls.append({
            "label": kwargs.get("label"),
            "color": kwargs.get("color"),
            "linestyle": kwargs.get("linestyle"),
            "marker": kwargs.get("marker"),
            "markersize": kwargs.get("markersize"),
        })
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture)

    df = pd.DataFrame(
        [
            {"param": "d", "param_value": 10.0, "shadow": "naive", "dispatch": "greedy", "mean_savings": 1.0},
            {"param": "d", "param_value": 10.0, "shadow": "pb", "dispatch": "greedy", "mean_savings": 1.5},
            {"param": "d", "param_value": 10.0, "shadow": "naive", "dispatch": "batch", "mean_savings": 0.8},
            {"param": "d", "param_value": 10.0, "shadow": "naive", "dispatch": "rbatch", "mean_savings": 0.9},
            {"param": "d", "param_value": 10.0, "shadow": "opt", "dispatch": "opt", "mean_savings": 2.0},
        ]
    )

    out_path = tmp_path / "plot.png"
    plot_results._plot_metric_sweep(
        df,
        "mean_savings",
        include=None,
        title=None,
        csv_hint=None,
        out_arg=str(out_path),
    )

    labels = [call["label"] for call in calls]
    assert labels == ["NAIVE + GRE", "PB + GRE", "NAIVE + BAT", "NAIVE + RBAT", "OPT"]

    greedy_color = mcolors.to_rgb(calls[0]["color"])
    batch_color = mcolors.to_rgb(calls[2]["color"])
    rbatch_color = mcolors.to_rgb(calls[3]["color"])

    assert all(batch <= greedy for batch, greedy in zip(batch_color, greedy_color))
    assert any(batch < greedy for batch, greedy in zip(batch_color, greedy_color))
    assert all(rbatch <= greedy for rbatch, greedy in zip(rbatch_color, greedy_color))
    assert any(rbatch < greedy for rbatch, greedy in zip(rbatch_color, greedy_color))
    assert all(batch <= rbatch for batch, rbatch in zip(batch_color, rbatch_color))

    assert calls[0]["linestyle"] == "-"
    assert calls[2]["linestyle"] == ":"
    assert calls[3]["linestyle"] == "--"

    assert calls[0]["marker"] == "o"
    assert calls[1]["marker"] == "^"
    assert calls[4]["marker"] == "D"

    assert calls[0]["markersize"] < calls[3]["markersize"] < calls[2]["markersize"]

    opt_color = mcolors.to_rgb(calls[4]["color"])
    assert opt_color == (0.06666666666666667, 0.06666666666666667, 0.06666666666666667)


def test_hd_shadow_uses_dashdot_line_without_errorbar(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    errorbar_calls: list[dict[str, object]] = []
    plot_calls: list[dict[str, object]] = []

    original_errorbar = Axes.errorbar
    original_plot = Axes.plot

    def capture_errorbar(self, *args, **kwargs):
        errorbar_calls.append(kwargs)
        return original_errorbar(self, *args, **kwargs)

    def capture_plot(self, *args, **kwargs):
        plot_calls.append(kwargs)
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", capture_errorbar)
    monkeypatch.setattr(Axes, "plot", capture_plot)

    df = pd.DataFrame(
        [
            {
                "param": "d",
                "param_value": 10.0,
                "shadow": "hd",
                "dispatch": "greedy",
                "mean_savings": 1.1,
                "std_savings": 0.05,
                "trial_count": 10,
            }
        ]
    )

    out_path = tmp_path / "plot.png"
    plot_results._plot_metric_sweep(
        df,
        "mean_savings",
        include=None,
        title=None,
        csv_hint=None,
        out_arg=str(out_path),
    )

    assert errorbar_calls == []
    assert len(plot_calls) == 1

    call_kwargs = plot_calls[0]
    assert call_kwargs["linestyle"] == "-."
    assert call_kwargs["marker"] is None
    assert call_kwargs["label"] == "HD + GRE"


def test_ratio_metric_clamps_upper_ylim(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    original_subplots = plot_results.plt.subplots

    def wrapped_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plot_results.plt, "subplots", wrapped_subplots)

    df = pd.DataFrame(
        [
            {
                "param": "d",
                "param_value": 5.0,
                "shadow": "naive",
                "dispatch": "greedy",
                "mean_ratio_opt": 0.95,
            }
        ]
    )

    out_path = tmp_path / "plot.png"
    plot_results._plot_metric_sweep(
        df,
        "mean_ratio_opt",
        include=None,
        title=None,
        csv_hint=None,
        out_arg=str(out_path),
    )

    ax = captured["ax"]
    _, ymax = ax.get_ylim()
    assert ymax == 1.0
