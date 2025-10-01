"""H3-based type mapping helpers for average-dual tables.

This module mirrors :mod:`ddp.mappings.uniform_grid` but hashes job origin and
destination coordinates onto the discrete H3 grid.  The mapping returns a pair
of hexadecimal strings representing the sender and recipient cells at the
configured resolution.  These type keys integrate with the average-dual
pipeline so empirical dual estimates can be averaged by location pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
try:
    import h3  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - environment without h3
    from ddp._vendor import h3lite as h3  # type: ignore[assignment]

from ddp.model import Job

TypeKey = tuple[str, str]

__all__ = [
    "H3PairMapping",
    "DEFAULT_RESOLUTION",
    "mapping",
    "make_mapping",
    "job_mapping",
]


_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


def _cell_from_latlng(lat: float, lng: float, resolution: int) -> str:
    """Return the H3 cell for the given ``lat``/``lng`` pair."""

    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lng, resolution)
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    msg = "Compatible h3 API not available: expected geo_to_h3 or latlng_to_cell"
    raise AttributeError(msg)


@dataclass(**_DATACLASS_KWARGS)
class H3PairMapping:
    """Quantise origin/destination points into H3 cells."""

    resolution: int

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, int):
            msg = "H3 resolution must be an integer"
            raise TypeError(msg)
        if not 0 <= self.resolution <= 15:
            msg = "H3 resolution must fall within the inclusive range [0, 15]"
            raise ValueError(msg)

    def __call__(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
    ) -> TypeKey:
        """Return the H3 cells for the origin and destination."""

        origin_hex = _cell_from_latlng(origin_lat, origin_lng, self.resolution)
        dest_hex = _cell_from_latlng(dest_lat, dest_lng, self.resolution)
        return origin_hex, dest_hex


DEFAULT_RESOLUTION = 8

mapping = H3PairMapping(resolution=DEFAULT_RESOLUTION)


def make_mapping(resolution: int) -> H3PairMapping:
    """Return an :class:`H3PairMapping` configured for ``resolution``."""

    return H3PairMapping(resolution=resolution)


def job_mapping(job: Job, mapper: H3PairMapping = mapping) -> TypeKey:
    """Return the H3 cell pair for ``job`` using ``mapper``."""

    origin_lat, origin_lng = job.origin
    dest_lat, dest_lng = job.dest
    return mapper(origin_lat, origin_lng, dest_lat, dest_lng)

