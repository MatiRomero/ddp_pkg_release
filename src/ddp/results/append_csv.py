"""Utilities for safely appending CSV data from multiple workers."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import fcntl


def append_csv_locked(src_path: str, dst_path: str, lock_path: Optional[str] = None) -> None:
    """Append rows from ``src_path`` into ``dst_path`` with an inter-process lock.

    The function reads the ``src_path`` CSV, separating its header row from data
    rows. It then acquires an exclusive ``fcntl`` lock on ``lock_path`` (which
    defaults to ``dst_path + ".lock"``) before appending the data to ``dst_path``.
    If the destination file does not yet exist, the header row is written
    alongside the data. The destination is flushed and ``os.fsync`` is called
    before releasing the lock to ensure durability.
    """

    src = Path(src_path)
    dst = Path(dst_path)
    if lock_path is None:
        lock = Path(f"{dst_path}.lock")
    else:
        lock = Path(lock_path)

    if not src.exists():
        raise FileNotFoundError(f"Source CSV not found: {src}")

    with src.open(newline="") as src_file:
        reader = csv.reader(src_file)
        try:
            header = next(reader)
        except StopIteration:
            # Empty source file – nothing to append.
            return
        data_rows = list(reader)

    # Ensure the destination directory (and lock file directory) exist before
    # acquiring the lock so workers don't race on directory creation.
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)

    with lock.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            dst_exists = dst.exists()
            mode = "a" if dst_exists else "w"
            with dst.open(mode, newline="") as dst_file:
                writer = csv.writer(dst_file)
                if dst_exists:
                    if data_rows:
                        writer.writerows(data_rows)
                else:
                    writer.writerow(header)
                    if data_rows:
                        writer.writerows(data_rows)
                dst_file.flush()
                os.fsync(dst_file.fileno())
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

