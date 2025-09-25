import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ddp.scripts import build_average_duals, inspect_average_duals  # noqa: E402
from ddp.scripts.average_duals import _load_mapping  # noqa: E402


class InspectAverageDualsTest(unittest.TestCase):
    def test_summarises_stats_and_heatmap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            hd_path = tmp_path / "hd.csv"
            hd_path.write_text(
                "origin_x,origin_y,dest_x,dest_y,hindsight_dual\n"
                "0.12,0.14,0.47,0.52,1.0\n"
                "0.18,0.19,0.48,0.51,3.0\n"
                "1.05,-0.95,-0.45,0.38,2.0\n"
            )

            mapping, _expected = _load_mapping("ddp.mappings.uniform_grid:mapping")
            stats = build_average_duals.compute_average_duals(hd_path, mapping)

            rows = inspect_average_duals._summarise_stats(stats)
            self.assertEqual(len(rows), 2)
            keys = [row[0] for row in rows]
            self.assertIn("((1, 1), (4, 5))", keys)
            self.assertIn("((10, -10), (-5, 3))", keys)
            row_map = {row[0]: row for row in rows}
            self.assertAlmostEqual(row_map["((1, 1), (4, 5))"][2], 2.0)
            self.assertAlmostEqual(row_map["((1, 1), (4, 5))"][3], 1.0)
            self.assertAlmostEqual(row_map["((10, -10), (-5, 3))"][2], 2.0)
            self.assertAlmostEqual(row_map["((10, -10), (-5, 3))"][3], 0.0)

            heatmap, x_ticks, y_ticks = inspect_average_duals._make_origin_heatmap(stats)
            self.assertEqual(heatmap.shape, (len(y_ticks), len(x_ticks)))
            self.assertGreater(heatmap.sum(), 0)


if __name__ == "__main__":
    unittest.main()
