from __future__ import annotations

import sys
from pathlib import Path

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
