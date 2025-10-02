import numpy as np

from ddp.model import Job
from ddp.scripts.run import AverageDualError, _load_precomputed_ad_shadows


def _make_jobs(count: int) -> list[Job]:
    return [
        Job(origin=(float(i), float(i + 1)), dest=(float(i + 2), float(i + 3)), timestamp=float(i))
        for i in range(count)
    ]


def test_loads_from_sequence() -> None:
    jobs = _make_jobs(3)
    table = [1.0, 2.5, -0.5]

    result = _load_precomputed_ad_shadows(jobs, table)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array(table, dtype=float))


def test_loads_from_mapping_with_string_keys() -> None:
    jobs = _make_jobs(2)
    table = {"0": 4.2, "1": -1.1}

    result = _load_precomputed_ad_shadows(jobs, table)

    np.testing.assert_allclose(result, np.array([4.2, -1.1], dtype=float))


def test_missing_entries_raise_error() -> None:
    jobs = _make_jobs(2)
    table = {"0": 3.0}

    try:
        _load_precomputed_ad_shadows(jobs, table)
    except AverageDualError as exc:
        assert "Missing average-dual values" in str(exc)
    else:  # pragma: no cover - defensive, should not happen
        raise AssertionError("Expected AverageDualError when entries are missing")
