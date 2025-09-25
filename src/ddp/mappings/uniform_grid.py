"""Uniform spatial grid mapping helpers for average-dual tables.

The :class:`UniformGridMapping` quantises origin/destination coordinates onto a
regular lattice whose spacing is controlled by ``type_width``. It returns a
hashable type key ``((o_ix, o_iy), (d_ix, d_iy))`` where each component is the
integer index of the snapped cell. When origin/destination bounds are provided
the mapping pre-computes the Cartesian product of reachable grid indices and
exposes the resulting ``expected_types`` set so average-dual tooling can report
coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import math
from typing import Iterator, Sequence

from ddp.model import Job

GridIndex = tuple[int, int]
TypeKey = tuple[GridIndex, GridIndex]

__all__ = [
    "UniformGridMapping",
    "DEFAULT_WIDTH",
    "FINE_WIDTH",
    "COARSE_WIDTH",
    "mapping",
    "fine_mapping",
    "coarse_mapping",
    "make_mapping",
    "job_mapping",
]


# ``slots`` support for dataclasses was introduced in Python 3.10.
_DATACLASS_KWARGS = {"slots": True} if sys.version_info >= (3, 10) else {}


@dataclass(**_DATACLASS_KWARGS)
class UniformGridMapping:
    """Quantise origin/destination points onto a uniform spatial grid."""

    type_width: float
    origin_offset: tuple[float, float] = (0.0, 0.0)
    dest_offset: tuple[float, float] = (0.0, 0.0)
    origin_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    dest_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None
    expected_types: set[TypeKey] | None = None

    def __post_init__(self) -> None:
        if self.type_width <= 0:
            raise ValueError("type_width must be positive")
        self.origin_offset = tuple(float(v) for v in self.origin_offset)
        self.dest_offset = tuple(float(v) for v in self.dest_offset)
        if self.expected_types is not None:
            self.expected_types = set(self.expected_types)
        else:
            self.expected_types = self._compute_expected_types()

    def _snap(self, x: float, y: float, *, offsets: Sequence[float]) -> GridIndex:
        """Return the grid cell indices covering ``(x, y)``."""

        width = self.type_width
        ix = math.floor((x - offsets[0]) / width)
        iy = math.floor((y - offsets[1]) / width)
        return (ix, iy)

    def __call__(self, origin_x: float, origin_y: float, dest_x: float, dest_y: float) -> TypeKey:
        """Return the snapped origin/destination grid indices."""

        origin_index = self._snap(origin_x, origin_y, offsets=self.origin_offset)
        dest_index = self._snap(dest_x, dest_y, offsets=self.dest_offset)
        return origin_index, dest_index

    def _compute_expected_types(self) -> set[TypeKey] | None:
        if self.origin_bounds is None or self.dest_bounds is None:
            return None

        origin_range = tuple(self._iter_indices(self.origin_bounds, self.origin_offset))
        dest_range = tuple(self._iter_indices(self.dest_bounds, self.dest_offset))
        expected: set[TypeKey] = set()
        for origin_index in origin_range:
            for dest_index in dest_range:
                expected.add((origin_index, dest_index))
        return expected

    def _iter_indices(
        self,
        bounds: tuple[tuple[float, float], tuple[float, float]],
        offsets: Sequence[float],
    ) -> Iterator[GridIndex]:
        (x_min, x_max), (y_min, y_max) = bounds
        x_indices = self._index_range(x_min, x_max, offsets[0])
        y_indices = self._index_range(y_min, y_max, offsets[1])
        for ix in x_indices:
            for iy in y_indices:
                yield (ix, iy)

    def _index_range(self, low: float, high: float, offset: float) -> range:
        if low > high:
            low, high = high, low
        width = self.type_width
        start = math.floor((low - offset) / width)
        stop = math.floor((high - offset) / width)
        if stop < start:
            start, stop = stop, start
        return range(start, stop + 1)


DEFAULT_WIDTH = 0.1
FINE_WIDTH = 0.01
COARSE_WIDTH = 0.2

mapping = UniformGridMapping(type_width=DEFAULT_WIDTH)
fine_mapping = UniformGridMapping(type_width=FINE_WIDTH)
coarse_mapping = UniformGridMapping(type_width=COARSE_WIDTH)

def make_mapping(type_width: float) -> UniformGridMapping:
    """Return a ``UniformGridMapping`` configured for ``type_width``."""

    return UniformGridMapping(type_width=type_width)

def job_mapping(job: Job, mapper: UniformGridMapping = mapping) -> TypeKey:
    """Return the grid type for ``job`` using ``mapper`` (default: ``mapping``)."""

    origin_x, origin_y = job.origin
    dest_x, dest_y = job.dest
    return mapper(origin_x, origin_y, dest_x, dest_y)
