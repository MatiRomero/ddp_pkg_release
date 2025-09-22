from __future__ import annotations

from io import StringIO
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ddp.model import Job  # noqa: E402
from ddp.scripts import load_jobs_from_csv  # noqa: E402


def _jobs_from_csv(content: str, **kwargs) -> list[Job]:
    return load_jobs_from_csv(StringIO(content), **kwargs)


def test_load_jobs_happy_path() -> None:
    csv_content = """sender_lat,sender_lng,recipient_lat,recipient_lng,platform_order_time
30.1,120.2,30.2,120.3,2023-07-01T12:00:00
31.0,121.0,31.1,121.1,2023-07-01T12:00:10
"""

    jobs = _jobs_from_csv(csv_content)

    assert len(jobs) == 2
    assert jobs[0].origin == (30.1, 120.2)
    assert jobs[0].dest == (30.2, 120.3)
    assert jobs[0].timestamp == 0.0
    assert jobs[1].timestamp == pytest.approx(10.0)


def test_load_jobs_with_alternate_timestamp_column() -> None:
    csv_content = """sender_lat,sender_lng,recipient_lat,recipient_lng,custom_time
30.1,120.2,30.2,120.3,2023-07-01T12:00:00
"""

    jobs = _jobs_from_csv(csv_content, timestamp_column="custom_time")

    assert len(jobs) == 1
    assert jobs[0].timestamp == 0.0


def test_jobs_sorted_by_timestamp_and_row_order() -> None:
    csv_content = """sender_lat,sender_lng,recipient_lat,recipient_lng,platform_order_time
30.1,120.2,30.2,120.3,2023-07-01T12:00:05
30.3,120.4,30.4,120.5,2023-07-01T12:00:00
30.5,120.6,30.6,120.7,2023-07-01T12:00:00
"""

    jobs = _jobs_from_csv(csv_content)

    assert [job.origin for job in jobs] == [
        (30.3, 120.4),
        (30.5, 120.6),
        (30.1, 120.2),
    ]
    assert jobs[0].timestamp == 0.0
    assert jobs[1].timestamp == 0.0
    assert jobs[2].timestamp == pytest.approx(5.0)


def test_rows_missing_required_fields_are_ignored() -> None:
    csv_content = """sender_lat,sender_lng,recipient_lat,recipient_lng,platform_order_time
,,30.2,120.3,2023-07-01T12:00:00
31.0,121.0,31.1,121.1,2023-07-01T12:00:10
"""

    jobs = _jobs_from_csv(csv_content)

    assert len(jobs) == 1
    assert jobs[0].origin == (31.0, 121.0)


def test_missing_required_columns_raise_value_error() -> None:
    csv_content = """sender_lat,sender_lng,recipient_lat,platform_order_time
30.1,120.2,30.2,2023-07-01T12:00:00
"""

    with pytest.raises(ValueError, match="missing required columns"):
        _jobs_from_csv(csv_content)

