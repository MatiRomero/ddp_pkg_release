from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ddp.model import Job
from ddp.mappings.h3_pairs import H3PairMapping, job_mapping
from ddp.scripts.run import _export_runtime_ad_updates, _resolve_average_duals

try:
    import h3  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without h3
    from ddp._vendor import h3lite as h3  # type: ignore[assignment]


def _geo_to_h3(lat: float, lng: float, resolution: int) -> str:
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lng, resolution)
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lng, resolution)
    raise AttributeError("Compatible h3 geo->cell function not available")


def _h3_to_geo(cell: str) -> tuple[float, float]:
    if hasattr(h3, "h3_to_geo"):
        return h3.h3_to_geo(cell)
    if hasattr(h3, "cell_to_latlng"):
        lat, lng = h3.cell_to_latlng(cell)
        return float(lat), float(lng)
    raise AttributeError("Compatible h3 cell->geo function not available")


def _make_job_for_hex(pair: tuple[str, str]) -> Job:
    sender_hex, recipient_hex = pair
    sender_lat, sender_lng = _h3_to_geo(sender_hex)
    recipient_lat, recipient_lng = _h3_to_geo(recipient_hex)
    return Job(origin=(sender_lat, sender_lng), dest=(recipient_lat, recipient_lng), timestamp=0.0)


def test_resolve_average_duals_prefers_neighbors(tmp_path: Path) -> None:
    mapping = H3PairMapping(resolution=8)

    sender_hex = _geo_to_h3(0.0, 0.0, mapping.resolution)
    recipient_hex = _geo_to_h3(0.01, 0.01, mapping.resolution)
    job = _make_job_for_hex((sender_hex, recipient_hex))

    if hasattr(h3, "grid_disk"):
        neighborhood = h3.grid_disk(sender_hex, 1)
    elif hasattr(h3, "k_ring"):
        neighborhood = h3.k_ring(sender_hex, 1)
    else:  # pragma: no cover - unexpected binding
        raise AttributeError("Compatible h3 neighborhood function not available")
    sender_neighbors = [cell for cell in neighborhood if cell != sender_hex]
    assert sender_neighbors, "expected at least one neighbouring sender hex"
    neighbor_sender = sender_neighbors[0]

    neighbor_value = 2.5
    ad_table = {str((neighbor_sender, recipient_hex)): neighbor_value}
    duals = np.array([7.0], dtype=float)

    sp, missing, assignments, updated_table = _resolve_average_duals(
        [job],
        duals,
        ad_table,
        lambda j: job_mapping(j, mapping),
        missing="neighbor",
    )

    assert math.isclose(sp[0], neighbor_value)
    assert missing == []
    assert str(job_mapping(job, mapping)) in updated_table

    assignment = assignments[0]
    assert assignment.source == "neighbor"
    assert assignment.neighbor_radius == 1
    assert assignment.neighbor_source_type == str((neighbor_sender, recipient_hex))

    lookup_path = tmp_path / "meituan_ad_day0_d10_res8_lookup.csv"
    lookup_path.write_text("type,mean_dual\n")

    _export_runtime_ad_updates(
        ad_path=str(lookup_path),
        jobs=[job],
        assignments=assignments,
        table=updated_table,
    )

    base = lookup_path.parent / "meituan_ad_day0_d10_res8"
    lookup_runtime = base.with_name(base.name + "_lookup_runtime.csv")
    summary_runtime = base.with_name(base.name + "_summary_runtime.csv")
    full_runtime = base.with_name(base.name + "_full_runtime.csv")

    assert lookup_runtime.exists()
    assert summary_runtime.exists()
    assert full_runtime.exists()

    with lookup_runtime.open() as handle:
        rows = list(csv.DictReader(handle))
    types = {row["type"] for row in rows}
    assert str(job_mapping(job, mapping)) in types

    with summary_runtime.open() as handle:
        summary_rows = list(csv.DictReader(handle))
    assert any(row["source"] == "neighbor" for row in summary_rows)

    with full_runtime.open() as handle:
        full_rows = list(csv.DictReader(handle))
    assert full_rows[0]["source"] == "neighbor"
    assert full_rows[0]["neighbor_source"] == str((neighbor_sender, recipient_hex))
