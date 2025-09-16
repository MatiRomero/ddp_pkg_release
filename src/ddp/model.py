"""Core data structures and helpers for dynamic delivery pooling jobs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Sequence

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class Job:
    """Represents a delivery job with an origin, destination, and timestamp."""

    origin: Point
    dest: Point
    timestamp: float

    @property
    def length(self) -> float:
        """Return the direct travel distance for the job."""

        return distance(self.origin, self.dest)


def distance(a: Point, b: Point, *, ord: float | int | None = 2) -> float:
    """Return the vector norm between two 2-D points."""

    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.linalg.norm(delta, ord=ord))


def _as_point(pair: Sequence[float]) -> Point:
    """Convert a numeric pair into a :class:`Point` tuple."""

    if len(pair) != 2:
        msg = "Points must be two-dimensional"
        raise ValueError(msg)
    return float(pair[0]), float(pair[1])


def generate_jobs(n: int, rng: np.random.Generator) -> list[Job]:
    """Generate ``n`` random jobs using the provided RNG with timestamps starting at 1."""

    if n <= 1:
        msg = "Number of jobs must exceed one"
        raise ValueError(msg)

    origins = rng.random((n, 2))
    destinations = rng.random((n, 2))
    timestamps = np.arange(1, n + 1, dtype=float)
    return [
        Job(
            origin=_as_point(origin),
            dest=_as_point(destination),
            timestamp=float(ts),
        )
        for origin, destination, ts in zip(origins, destinations, timestamps)
    ]


def distances_along_path(
    points: Iterable[Point], *, ord: float | int | None = 2
) -> list[float]:
    """Compute cumulative distances along a path defined by ``points``."""

    iterator = iter(points)
    try:
        previous = next(iterator)
    except StopIteration:
        return []

    cumulative = 0.0
    distances: list[float] = []
    for point in iterator:
        cumulative += distance(previous, point, ord=ord)
        distances.append(cumulative)
        previous = point
    return distances


def reward(jobs: Sequence[Job], *, ord: float | int | None = 2) -> float:
    """Compute the pooling reward for a collection of jobs."""

    job_list = list(jobs)
    if not job_list:
        return 0.0

    total_length = sum(distance(job.origin, job.dest, ord=ord) for job in job_list)
    origins = [job.origin for job in job_list]
    destinations = [job.dest for job in job_list]

    min_distance = float("inf")
    for origin_order in permutations(origins):
        for dest_order in permutations(destinations):
            path = [*origin_order, *dest_order]
            cumulative = distances_along_path(path, ord=ord)
            path_length = cumulative[-1] if cumulative else 0.0
            if path_length < min_distance:
                min_distance = path_length

    if min_distance == float("inf"):
        min_distance = 0.0
    return total_length - min_distance
