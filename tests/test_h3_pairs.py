from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ddp.mappings import h3_pairs
from ddp.scripts import meituan_average_duals


def test_cell_from_latlng_falls_back_to_modern_api() -> None:
    sentinel = "abc123"
    latlng_to_cell = mock.Mock(return_value=sentinel)
    fake_h3 = SimpleNamespace(latlng_to_cell=latlng_to_cell)

    with mock.patch.object(h3_pairs, "h3", fake_h3):
        result = h3_pairs._cell_from_latlng(10.0, 20.0, 5)

    assert result == sentinel
    latlng_to_cell.assert_called_once_with(10.0, 20.0, 5)


def test_neighbor_pairs_falls_back_to_grid_disk() -> None:
    hex_id = "abc"
    neighbors = [[hex_id], ["def", "ghi"]]
    grid_disk = mock.Mock(return_value=neighbors)
    fake_h3 = SimpleNamespace(grid_disk=grid_disk)

    with mock.patch.object(meituan_average_duals, "h3", fake_h3):
        result = meituan_average_duals._neighbor_pairs(hex_id, 2)

    assert result == {hex_id, "def", "ghi"}
    grid_disk.assert_called_once_with(hex_id, 2)


def test_neighbor_pairs_handles_flat_grid_disk_response() -> None:
    hex_id = "abc"
    neighbors = [hex_id, "def"]
    grid_disk = mock.Mock(return_value=neighbors)
    fake_h3 = SimpleNamespace(grid_disk=grid_disk)

    with mock.patch.object(meituan_average_duals, "h3", fake_h3):
        result = meituan_average_duals._neighbor_pairs(hex_id, 1)

    assert result == {hex_id, "def"}
    grid_disk.assert_called_once_with(hex_id, 1)
