import csv
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ddp.mappings.uniform_grid import job_mapping as uniform_job_mapping  # noqa: E402
from ddp.model import Job  # noqa: E402
from ddp.scripts import average_duals, build_average_duals  # noqa: E402
from ddp.scripts.run import load_average_dual_mapper, load_average_duals  # noqa: E402


class BuildAverageDualsTest(unittest.TestCase):
    def test_builds_average_dual_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            hd_path = tmp_path / "hd.csv"
            rows = [
                {
                    "origin_x": "0.12",
                    "origin_y": "0.14",
                    "dest_x": "0.47",
                    "dest_y": "0.52",
                    "hindsight_dual": "1.0",
                },
                {
                    "origin_x": "0.18",
                    "origin_y": "0.19",
                    "dest_x": "0.48",
                    "dest_y": "0.51",
                    "hindsight_dual": "3.0",
                },
                {
                    "origin_x": "1.05",
                    "origin_y": "-0.95",
                    "dest_x": "-0.45",
                    "dest_y": "0.38",
                    "hindsight_dual": "2.0",
                },
            ]
            with hd_path.open("w", newline="") as handle:
                fieldnames = list(rows[0])
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            mapping, _expected = average_duals._load_mapping("ddp.mappings.uniform_grid:mapping")
            stats = build_average_duals.compute_average_duals(hd_path, mapping)
            out_path = tmp_path / "average_duals.csv"
            build_average_duals.write_average_duals(out_path, stats)

            with out_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["type", "mean_dual", "std_dev", "count"])
                output_rows = list(reader)

            self.assertEqual(len(output_rows), 2)
            rows_by_type = {row["type"]: row for row in output_rows}
            type_a = "((1, 1), (4, 5))"
            type_b = "((10, -10), (-5, 3))"
            self.assertIn(type_a, rows_by_type)
            self.assertIn(type_b, rows_by_type)
            self.assertEqual(rows_by_type[type_a]["count"], "2")
            self.assertEqual(rows_by_type[type_b]["count"], "1")
            self.assertAlmostEqual(float(rows_by_type[type_a]["mean_dual"]), 2.0)
            self.assertAlmostEqual(float(rows_by_type[type_b]["mean_dual"]), 2.0)
            self.assertAlmostEqual(float(rows_by_type[type_a]["std_dev"]), 1.0)
            self.assertAlmostEqual(float(rows_by_type[type_b]["std_dev"]), 0.0)

            table = load_average_duals(str(out_path))
            mapper = load_average_dual_mapper("ddp.mappings.uniform_grid:job_mapping")
            jobs = [
                Job(origin=(0.12, 0.18), dest=(0.47, 0.52), timestamp=0.0),
                Job(origin=(1.08, -0.92), dest=(-0.42, 0.38), timestamp=1.0),
            ]
            mapped_types = [mapper(job) for job in jobs]
            self.assertEqual(mapped_types[0], uniform_job_mapping(jobs[0]))
            self.assertEqual(mapped_types[1], uniform_job_mapping(jobs[1]))
            self.assertAlmostEqual(table[str(mapped_types[0])], 2.0)
            self.assertAlmostEqual(table[str(mapped_types[1])], 2.0)


if __name__ == "__main__":
    unittest.main()
