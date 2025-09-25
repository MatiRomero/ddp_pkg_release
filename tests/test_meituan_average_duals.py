import pathlib
import sys
import tempfile
import unittest
from unittest import mock

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - handled via skip
    pd = None  # type: ignore[assignment]

try:
    import h3
except ModuleNotFoundError:  # pragma: no cover - handled via skip
    h3 = None  # type: ignore[assignment]

H3_AVAILABLE = h3 is not None and hasattr(h3, "geo_to_h3")


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from ddp.mappings.h3_pairs import make_mapping  # noqa: E402
from ddp.scripts.meituan_average_duals import (  # noqa: E402
    add_h3_columns,
    aggregate_by_hex,
    ensure_hd_cache,
    match_average_duals,
)


@unittest.skipIf(pd is None or not H3_AVAILABLE, "pandas and h3 are required for these tests")
class MeituanAverageDualsTest(unittest.TestCase):
    def test_add_h3_columns_and_aggregate(self) -> None:
        mapper = make_mapping(5)
        frame = pd.DataFrame(
            [
                {
                    "sender_lat": 39.90,
                    "sender_lng": 116.40,
                    "recipient_lat": 39.92,
                    "recipient_lng": 116.42,
                    "hindsight_dual": 2.0,
                },
                {
                    "sender_lat": 39.90,
                    "sender_lng": 116.40,
                    "recipient_lat": 39.92,
                    "recipient_lng": 116.42,
                    "hindsight_dual": 4.0,
                },
                {
                    "sender_lat": 39.85,
                    "sender_lng": 116.35,
                    "recipient_lat": 39.88,
                    "recipient_lng": 116.33,
                    "hindsight_dual": 1.0,
                },
            ]
        )

        enriched = add_h3_columns(frame, mapper)
        self.assertIn("sender_hex", enriched.columns)
        self.assertIn("recipient_hex", enriched.columns)

        summary = aggregate_by_hex(enriched)
        self.assertEqual(set(summary.columns), {"sender_hex", "recipient_hex", "mean_dual", "std_dual", "count"})
        self.assertEqual(summary.loc[summary["count"] == 2, "mean_dual"].iloc[0], 3.0)
        self.assertAlmostEqual(summary.loc[summary["count"] == 2, "std_dual"].iloc[0], 1.0)
        self.assertEqual(sorted(summary["count"]), [1, 2])

    def test_neighbor_fallback_and_missing_policy(self) -> None:
        mapper = make_mapping(5)
        base_sender, base_recipient = mapper(39.90, 116.40, 39.92, 116.42)
        neighbor_sender = next(iter(h3.k_ring(base_sender, 1) - {base_sender}))
        summary = pd.DataFrame(
            [
                {
                    "sender_hex": base_sender,
                    "recipient_hex": base_recipient,
                    "mean_dual": 3.5,
                    "std_dual": 0.0,
                    "count": 4,
                }
            ]
        )

        target = pd.DataFrame(
            [
                {
                    "sender_hex": neighbor_sender,
                    "recipient_hex": base_recipient,
                    "hindsight_dual": 9.0,
                },
                {
                    "sender_hex": "871f3d3fffffff",
                    "recipient_hex": "871f3d7fffffff",
                    "hindsight_dual": 5.0,
                },
            ]
        )

        enriched = match_average_duals(
            target,
            summary,
            neighbor_radius=1,
            missing_policy="hd",
        )

        self.assertEqual(enriched.loc[0, "ad_source"], "neighbor")
        self.assertEqual(enriched.loc[0, "ad_mean"], 3.5)
        self.assertEqual(enriched.loc[1, "ad_source"], "hd_fallback")
        self.assertEqual(enriched.loc[1, "ad_mean"], 5.0)

        zeroed = match_average_duals(
            target.iloc[[1]],
            summary,
            neighbor_radius=0,
            missing_policy="zero",
        )
        self.assertEqual(zeroed.loc[0, "ad_source"], "zero_fallback")
        self.assertEqual(zeroed.loc[0, "ad_mean"], 0.0)

    def test_ensure_hd_cache_uses_deadline_specific_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = pathlib.Path(tmpdir) / "cache"
            snapshot = pathlib.Path(tmpdir) / "snapshot.csv"
            snapshot.write_text("dummy")

            legacy_path = cache_dir / "day3_hd.csv"
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            legacy_path.write_text("legacy")

            frame = pd.DataFrame(
                [
                    {
                        "day": 3,
                        "job_index": 0,
                        "timestamp": 0.0,
                        "sender_lat": 0.0,
                        "sender_lng": 0.0,
                        "recipient_lat": 0.0,
                        "recipient_lng": 0.0,
                        "hindsight_dual": 1.0,
                    }
                ]
            )

            with mock.patch(
                "ddp.scripts.meituan_average_duals.compute_day_hd_duals",
                return_value=frame,
            ) as compute_mock:
                result = ensure_hd_cache(
                    day=3,
                    snapshot=snapshot,
                    cache_dir=cache_dir,
                    timestamp_column="timestamp",
                    deadline=12.5,
                    force=False,
                )

            expected_path = cache_dir / "day3_d12.5_hd.csv"
            self.assertTrue(expected_path.exists())
            self.assertTrue(frame.equals(result))
            compute_mock.assert_called_once()
            self.assertTrue(legacy_path.exists())


if __name__ == "__main__":
    unittest.main()

