import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ddp.mappings.uniform_grid import (  # noqa: E402
    COARSE_WIDTH,
    DEFAULT_WIDTH,
    FINE_WIDTH,
    UniformGridMapping,
)


class UniformGridMappingTest(unittest.TestCase):
    def test_snap_rounds_to_nearest_cell(self) -> None:
        mapping = UniformGridMapping(type_width=DEFAULT_WIDTH)
        key = mapping(0.12, -0.21, 0.86, 0.98)
        self.assertEqual(key, ((0, 0), (2, 2)))

    def test_expected_types_from_bounds(self) -> None:
        mapping = UniformGridMapping(
            type_width=COARSE_WIDTH,
            origin_bounds=((0.0, 1.1), (-0.2, 0.8)),
            dest_bounds=((0.0, 0.0), (0.0, 0.0)),
        )
        expected = mapping.expected_types
        self.assertIsNotNone(expected)
        assert expected is not None  # narrow type
        self.assertEqual(
            expected,
            {
                ((0, 0), (0, 0)),
                ((1, 0), (0, 0)),
                ((0, 1), (0, 0)),
                ((1, 1), (0, 0)),
            },
        )

    def test_custom_expected_types_passthrough(self) -> None:
        custom = {
            ((0, 0), (0, 0)),
            ((0, 0), (1, 0)),
        }
        mapping = UniformGridMapping(type_width=FINE_WIDTH, expected_types=custom)
        self.assertEqual(mapping.expected_types, custom)


class MappingLoaderTest(unittest.TestCase):
    def test_load_mapping_by_spec(self) -> None:
        from ddp.scripts import average_duals

        mapper, expected = average_duals._load_mapping(
            "ddp.mappings.uniform_grid:mapping"
        )
        self.assertTrue(callable(mapper))
        self.assertEqual(mapper(0.0, 0.0, 0.51, 0.51), ((0, 0), (1, 1)))
        if expected is not None:
            self.assertIn(((0, 0), (0, 0)), expected)


if __name__ == "__main__":
    unittest.main()
