"""Utilities for loading :class:`~ddp.model.Job` instances from CSV files."""

from __future__ import annotations

import csv
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import TextIO

from ddp.model import Job


_DEFAULT_TIMESTAMP_COLUMN = "platform_order_time"
_REQUIRED_COORD_COLUMNS = (
    "sender_lat",
    "sender_lng",
    "recipient_lat",
    "recipient_lng",
)


_DATACLASS_KWARGS: dict[str, object] = {}
if sys.version_info >= (3, 10):
    _DATACLASS_KWARGS["slots"] = True


@dataclass(**_DATACLASS_KWARGS)
class _RowRecord:
    """Internal representation of a CSV row prior to conversion into a job."""

    index: int
    timestamp: datetime
    sender_lat: float
    sender_lng: float
    recipient_lat: float
    recipient_lng: float
    metadata: dict[str, str | None]


def load_jobs_from_csv(
    csv_file: os.PathLike[str] | str | TextIO,
    *,
    timestamp_column: str = _DEFAULT_TIMESTAMP_COLUMN,
) -> list[Job]:
    """Load jobs from a CSV file exported by the Meituan dataset.

    Parameters
    ----------
    csv_file:
        A path to the CSV file or an already opened text file object.
    timestamp_column:
        Name of the column containing ISO formatted timestamps. Defaults to
        ``"platform_order_time"``.

    Returns
    -------
    list[Job]
        The jobs sorted by ascending timestamp (ties broken by row order).

    Raises
    ------
    ValueError
        If the CSV header is missing any of the required columns.
    """

    if not timestamp_column:
        msg = "Timestamp column name must be provided"
        raise ValueError(msg)

    if isinstance(csv_file, (str, os.PathLike)):
        with open(csv_file, newline="", encoding="utf-8") as handle:
            return _load_jobs_from_stream(handle, timestamp_column=timestamp_column)

    return _load_jobs_from_stream(csv_file, timestamp_column=timestamp_column)


def _load_jobs_from_stream(
    stream: TextIO,
    *,
    timestamp_column: str,
) -> list[Job]:
    reader = csv.DictReader(stream)

    header = reader.fieldnames or []
    required_columns = set(_REQUIRED_COORD_COLUMNS) | {timestamp_column}
    missing_columns = required_columns.difference(header)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        msg = f"CSV is missing required columns: {columns}"
        raise ValueError(msg)

    rows: list[_RowRecord] = []
    metadata_columns = {"da_id", "job_id", "day", "dataset_id"}
    # Existing CSVs may already have an incidental area/day column. Only the
    # stable identity columns opt into the strict enriched-data contract.
    enriched = bool({"job_id", "dataset_id"}.intersection(header))
    if enriched and not metadata_columns.issubset(header):
        raise ValueError("Enriched CSV requires da_id, job_id, day and dataset_id")
    for index, row in enumerate(reader):
        try:
            record = _convert_row(row, index, timestamp_column)
        except ValueError:
            if enriched:
                raise ValueError(f"Invalid enriched job row {index}; refusing to drop metadata")
            continue
        if enriched and any(not (row.get(key) or "").strip() for key in metadata_columns):
            raise ValueError(f"Missing job metadata at row {index}")
        if enriched:
            record.metadata["original_timestamp"] = row[timestamp_column]
        rows.append(record)

    if not rows:
        return []

    rows.sort(key=lambda item: (item.timestamp, item.index))
    if enriched and len({row.metadata["job_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate job_id in enriched CSV")
    base_timestamp = rows[0].timestamp

    jobs: list[Job] = []
    for record in rows:
        delta = record.timestamp - base_timestamp
        jobs.append(
            Job(
                origin=(record.sender_lat, record.sender_lng),
                dest=(record.recipient_lat, record.recipient_lng),
                timestamp=float(delta.total_seconds()),
                **record.metadata,
            )
        )
    return jobs


def _convert_row(row: dict[str, str | None], index: int, timestamp_column: str) -> _RowRecord:
    values = {column: row.get(column) for column in _REQUIRED_COORD_COLUMNS}
    values[timestamp_column] = row.get(timestamp_column)

    if any(value is None or str(value).strip() == "" for value in values.values()):
        msg = "Row is missing required field"
        raise ValueError(msg)

    try:
        timestamp = _parse_timestamp(str(values[timestamp_column]))
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc

    try:
        sender_lat = float(values["sender_lat"])
        sender_lng = float(values["sender_lng"])
        recipient_lat = float(values["recipient_lat"])
        recipient_lng = float(values["recipient_lng"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid coordinate value") from exc

    if not all(math.isfinite(v) for v in (sender_lat, sender_lng, recipient_lat, recipient_lng)):
        raise ValueError("Coordinates must be finite")

    return _RowRecord(
        index=index,
        timestamp=timestamp,
        sender_lat=sender_lat,
        sender_lng=sender_lng,
        recipient_lat=recipient_lat,
        recipient_lng=recipient_lng,
        metadata={key: (row.get(key) or "").strip() or None for key in ("da_id", "job_id", "day", "dataset_id")},
    )


def _parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if not cleaned:
        msg = "Timestamp value cannot be empty"
        raise ValueError(msg)

    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    return datetime.fromisoformat(cleaned)
