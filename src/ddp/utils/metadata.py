"""Collect experiment metadata for reproducibility."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def collect_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect experiment metadata for reproducibility.
    
    Returns a JSON-serializable dictionary containing:
    - timestamp (UTC, ISO format)
    - git commit hash (if repo is a git repo; else "unknown")
    - python version
    - platform / OS
    - ddp package version (if available, else "unknown")
    - working directory
    - extra (merged in if provided)
    
    Parameters
    ----------
    extra : dict, optional
        Additional metadata to merge into the result.
        
    Returns
    -------
    dict
        JSON-serializable metadata dictionary.
    """
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd().resolve()),
    }
    
    # Try to get git commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        metadata["git_commit"] = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        metadata["git_commit"] = "unknown"
    
    # Try to get package version
    try:
        import importlib.metadata
        metadata["ddp_version"] = importlib.metadata.version("ddp")
    except Exception:
        try:
            # Fallback for older Python versions
            import pkg_resources
            metadata["ddp_version"] = pkg_resources.get_distribution("ddp").version
        except Exception:
            metadata["ddp_version"] = "unknown"
    
    # Merge extra metadata if provided
    if extra:
        metadata.update(extra)
    
    return metadata

