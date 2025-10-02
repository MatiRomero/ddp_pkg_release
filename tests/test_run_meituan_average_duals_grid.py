from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_runner_module() -> object:
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    script_path = root_dir / "scripts" / "run_meituan_average_duals_grid.py"
    spec = importlib.util.spec_from_file_location(
        "run_meituan_average_duals_grid", script_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise RuntimeError("Unable to load run_meituan_average_duals_grid module")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


runner = _load_runner_module()


class _FakeFrame:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def to_csv(self, path: Path | str, index: bool = False) -> None:  # pragma: no cover - signature parity
        Path(path).write_text(self._payload)


def test_default_export_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Path] = {}

    def _fake_build_average_duals(**kwargs):  # type: ignore[no-untyped-def]
        created["call"] = kwargs  # ensure the fake was invoked
        return SimpleNamespace(
            summary=_FakeFrame("summary"),
            lookup=_FakeFrame("lookup"),
            job_lookup=_FakeFrame("job"),
        )

    def _fake_save_summary_map(summary, path):  # type: ignore[no-untyped-def]
        Path(path).write_text("map")

    monkeypatch.setattr(runner, "build_average_duals", _fake_build_average_duals)
    monkeypatch.setattr(runner, "save_summary_map", _fake_save_summary_map)
    monkeypatch.setattr(
        runner,
        "export_job_aligned_duals_csv",
        lambda job_lookup, path: Path(path).write_text("job"),
    )

    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    input_dir = working_dir / "input"
    input_dir.mkdir()

    exit_code = runner.main(
        [
            "--days",
            "1",
            "--deadlines",
            "10",
            "--resolutions",
            "7",
            "--data-dir",
            str(input_dir),
            "--jobs-pattern",
            "dummy_day{day}.csv",
        ]
    )

    assert exit_code == 0
    assert "call" in created
    assert "missing_policy" not in created["call"]

    expected_stem = runner.DEFAULT_EXPORT_STEM.format(
        day=1, d=runner._format_deadline(10), r=7
    )
    summary_path = working_dir / runner.DEFAULT_EXPORT_BASE / f"{expected_stem}_summary.csv"
    lookup_path = working_dir / runner.DEFAULT_EXPORT_BASE / f"{expected_stem}_lookup.csv"
    job_path = working_dir / runner.DEFAULT_EXPORT_BASE / f"{expected_stem}_full.csv"

    assert summary_path.read_text() == "summary"
    assert lookup_path.read_text() == "lookup"
    assert job_path.read_text() == "job"
