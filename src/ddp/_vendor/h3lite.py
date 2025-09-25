"""Minimal fallback implementation mimicking the subset of the ``h3`` API used in tests."""

from __future__ import annotations

from typing import Iterable, Set


def _scale(resolution: int) -> float:
    return float(max(1, resolution * 10))


def _encode(resolution: int, lat_idx: int, lng_idx: int) -> str:
    return f"hex_{resolution}_{lat_idx}_{lng_idx}"


def _decode(index: str) -> tuple[int, int, int]:
    try:
        _prefix, res_str, lat_str, lng_str = index.split("_")
        return int(res_str), int(lat_str), int(lng_str)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid h3lite index: {index!r}") from exc


def geo_to_h3(lat: float, lng: float, resolution: int) -> str:
    scale = _scale(resolution)
    lat_idx = int(round(lat * scale))
    lng_idx = int(round(lng * scale))
    return _encode(resolution, lat_idx, lng_idx)


def h3_to_geo(index: str) -> tuple[float, float]:
    resolution, lat_idx, lng_idx = _decode(index)
    scale = _scale(resolution)
    return lat_idx / scale, lng_idx / scale


def k_ring(index: str, k: int) -> Set[str]:
    resolution, lat_idx, lng_idx = _decode(index)
    cells: Set[str] = set()
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            cells.add(_encode(resolution, lat_idx + dx, lng_idx + dy))
    return cells


def h3_to_geo_boundary(index: str, geo_json: bool = False) -> Iterable[Iterable[float]]:
    lat, lng = h3_to_geo(index)
    delta = 0.0005
    points = [
        (lat - delta, lng - delta),
        (lat - delta, lng + delta),
        (lat + delta, lng + delta),
        (lat + delta, lng - delta),
        (lat - delta, lng - delta),
    ]
    if geo_json:
        return [[float(a), float(b)] for a, b in points]
    return points

