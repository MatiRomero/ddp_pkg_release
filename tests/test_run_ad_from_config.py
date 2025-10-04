from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _pythonpath_env() -> dict[str, str]:
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else str(src_dir)
    return env


def test_run_ad_from_config_dry_run(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom_ad_config.csv"
    export_dir = tmp_path / "exports"

    env = _pythonpath_env()

    gen_cmd = [
        sys.executable,
        "-m",
        "ddp.scripts.generate_ad_config",
        "--config",
        str(config_path),
        "--days",
        "20210305",
        "--deadlines",
        "900",
        "--resolutions",
        "9",
        "--data-dir",
        "sample_data",
        "--jobs-pattern",
        "sample_day{day}.csv",
        "--export-dir",
        str(export_dir),
    ]
    subprocess.run(gen_cmd, check=True, env=env, cwd=Path(__file__).resolve().parents[1])

    monkeypatch.setenv("SGE_TASK_ID", "1")

    run_cmd = [
        sys.executable,
        "-m",
        "ddp.scripts.run_ad_from_config",
        "--config",
        str(config_path),
        "--dry-run",
        "--history-days",
        "20210301,20210302,20210303",
    ]
    result = subprocess.run(
        run_cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )

    stdout = result.stdout
    assert "--day 20210305" in stdout
    assert "--deadline 900" in stdout
    assert "--resolution 9" in stdout
    assert "--data-dir sample_data" in stdout
    assert "--jobs-pattern sample_day{day}.csv" in stdout
    assert f"--export-dir {export_dir}" in stdout
    assert "--history-days 20210301,20210302,20210303" in stdout
