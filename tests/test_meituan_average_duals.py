import os
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

H3_AVAILABLE = h3 is not None and (
    hasattr(h3, "geo_to_h3") or hasattr(h3, "latlng_to_cell")
)


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


from ddp.mappings.h3_pairs import make_mapping  # noqa: E402
from ddp.scripts.meituan_average_duals import (  # noqa: E402
    add_h3_columns,
    aggregate_by_hex,
    ensure_hd_cache,
    main,
    export_average_duals_csv,
    export_job_aligned_duals_csv,
    PipelineResult,
    _neighbor_pairs,
    match_average_duals,
)
from ddp.scripts.run import load_average_duals  # noqa: E402


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
                    "sender_lat": 39.30,
                    "sender_lng": 116.00,
                    "recipient_lat": 39.40,
                    "recipient_lng": 116.10,
                    "hindsight_dual": 1.0,
                },
            ]
        )

        enriched = add_h3_columns(frame, mapper)
        self.assertIn("sender_hex", enriched.columns)
        self.assertIn("recipient_hex", enriched.columns)

        summary = aggregate_by_hex(enriched)
        self.assertEqual(
            set(summary.columns),
            {"sender_hex", "recipient_hex", "type", "mean_dual", "std_dual", "count"},
        )
        for row in summary.itertuples(index=False):
            self.assertEqual(row.type, str((row.sender_hex, row.recipient_hex)))
        self.assertEqual(summary.loc[summary["count"] == 2, "mean_dual"].iloc[0], 3.0)
        self.assertAlmostEqual(summary.loc[summary["count"] == 2, "std_dual"].iloc[0], 1.0)
        self.assertEqual(sorted(summary["count"]), [1, 2])

    def test_export_average_duals_roundtrip(self) -> None:
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
                    "sender_lat": 39.30,
                    "sender_lng": 116.00,
                    "recipient_lat": 39.40,
                    "recipient_lng": 116.10,
                    "hindsight_dual": 1.0,
                },
            ]
        )

        enriched = add_h3_columns(frame, mapper)
        summary = aggregate_by_hex(enriched)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            csv_path = tmpdir_path / "ad_lookup.csv"
            job_csv_path = tmpdir_path / "ad_job_lookup.csv"

            export_average_duals_csv(summary, csv_path)
            export_job_aligned_duals_csv(
                pd.DataFrame(
                    [
                        {"job_index": 0, "mean_dual": summary.iloc[0]["mean_dual"]},
                        {"job_index": 1, "mean_dual": summary.iloc[1]["mean_dual"]},
                    ]
                ),
                job_csv_path,
            )

            expected = {row.type: float(row.mean_dual) for row in summary.itertuples(index=False)}

            csv_loaded = load_average_duals(str(csv_path))

            self.assertEqual(csv_loaded, expected)

            job_loaded = pd.read_csv(job_csv_path)
            self.assertEqual(list(job_loaded.columns), ["job_index", "mean_dual"])
            self.assertListEqual(job_loaded["job_index"].tolist(), [0, 1])

    def test_neighbor_search_and_zero_fallback(self) -> None:
        mapper = make_mapping(5)
        base_sender, base_recipient = mapper(39.90, 116.40, 39.92, 116.42)
        neighbor_sender = next(iter(_neighbor_pairs(base_sender, 1) - {base_sender}))
        missing_sender, missing_recipient = mapper(39.10, 116.20, 39.15, 116.25)
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
                    "sender_hex": missing_sender,
                    "recipient_hex": missing_recipient,
                    "hindsight_dual": 5.0,
                },
            ]
        )

        enriched = match_average_duals(
            target,
            summary,
            neighbor_radius=1,
        )

        self.assertEqual(enriched.loc[0, "ad_source"], "neighbor")
        self.assertEqual(enriched.loc[0, "ad_mean"], 3.5)
        self.assertEqual(enriched.loc[1, "ad_source"], "zero_fallback")
        self.assertEqual(enriched.loc[1, "ad_mean"], 0.0)

    def test_main_writes_default_exports(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "sender_hex": "abc",
                    "recipient_hex": "def",
                    "type": "('abc', 'def')",
                    "mean_dual": 1.5,
                    "std_dual": 0.2,
                    "count": 3,
                }
            ],
            columns=["sender_hex", "recipient_hex", "type", "mean_dual", "std_dual", "count"],
        )
        lookup_df = pd.DataFrame(
            [
                {"type": "('abc', 'def')", "mean_dual": 1.5},
            ],
            columns=["type", "mean_dual"],
        )
        job_lookup_df = pd.DataFrame(
            [
                {"job_index": 0, "mean_dual": 1.5},
            ],
            columns=["job_index", "mean_dual"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            cache_dir = tmp_path / "cache"
            exports_dir = tmp_path / "data" / "average_duals"

            pipeline_result = PipelineResult(
                summary=summary_df,
                lookup=lookup_df,
                job_lookup=job_lookup_df,
            )
            self.assertIs(pipeline_result.job_lookup, job_lookup_df)

            cwd = pathlib.Path.cwd()
            try:
                os.chdir(tmp_path)
                with mock.patch(
                    "ddp.scripts.meituan_average_duals.build_average_duals",
                    return_value=pipeline_result,
                ) as build_mock:
                    exit_code = main(
                        [
                            "--day",
                            "5",
                            "--data-dir",
                            str(tmp_path / "data"),
                            "--cache-dir",
                            str(cache_dir),
                        ]
                    )
            finally:
                os.chdir(cwd)

            self.assertEqual(exit_code, 0)
            build_mock.assert_called_once()

            stem = "meituan_ad_day5_d20_res8"
            summary_path = exports_dir / f"{stem}_summary.csv"
            ad_csv_path = exports_dir / f"{stem}_lookup.csv"
            ad_job_csv_path = exports_dir / f"{stem}_full.csv"

            self.assertTrue(summary_path.exists())
            self.assertTrue(ad_csv_path.exists())
            self.assertTrue(ad_job_csv_path.exists())

            summary_loaded = pd.read_csv(summary_path)
            pd.testing.assert_frame_equal(summary_loaded, summary_df, check_dtype=False)

            ad_lookup_loaded = pd.read_csv(ad_csv_path)
            self.assertEqual(list(ad_lookup_loaded.columns), ["type", "mean_dual"])
            self.assertEqual(ad_lookup_loaded.iloc[0]["type"], "('abc', 'def')")
            self.assertAlmostEqual(ad_lookup_loaded.iloc[0]["mean_dual"], 1.5)

            ad_job_lookup_loaded = pd.read_csv(ad_job_csv_path)
            self.assertEqual(list(ad_job_lookup_loaded.columns), ["job_index", "mean_dual"])
            pd.testing.assert_frame_equal(
                ad_job_lookup_loaded,
                job_lookup_df,
                check_dtype=False,
            )

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

