"""Exercise planning arithmetic, live-log races, and retrieval-integrity failures."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.artifact_inventory import create, verify
from tools.estimate_etc import estimate
from tools.watch_run import read_status, watch


class EstimateTests(unittest.TestCase):
    def test_complete_timing_and_aggregate_envelope(self):
        result = estimate({"optimizer_step_seconds": [2, 4, 6],
                           "full_validation_seconds": [10, 20], "setup_seconds": 100,
                           "checkpoint_seconds": 30, "transfer_seconds": 50},
                          steps=10, validation_passes=2, gpu_count=2,
                          hourly_gpu_price=3, maximum_hours=1, storage_cost=4)
        self.assertEqual(result["median_etc_seconds"], 250)
        self.assertEqual(result["p90_etc_seconds"], 280)
        self.assertEqual(result["maximum_total_cost"], 10)
        self.assertAlmostEqual(result["projected_compute_cost_median"], 250 / 3600 * 6)

    def test_no_validation_phase_and_duration_warning(self):
        result = estimate({"optimizer_step_seconds": [3600]}, steps=2,
                          validation_passes=0, hourly_gpu_price=1, maximum_hours=1)
        self.assertEqual(result["validation_samples"], 0)
        self.assertTrue(result["p90_scenario_exceeds_maximum_hours"])

    def test_reject_invalid_counts_prices_and_timings(self):
        defaults = dict(steps=1, validation_passes=0)
        for overrides in ({"steps": -1}, {"steps": 1.5}, {"gpu_count": 0},
                          {"hourly_gpu_price": -1}, {"storage_cost": float("inf")},
                          {"maximum_hours": 1}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                estimate({"optimizer_step_seconds": [1]}, **(defaults | overrides))
        for calibration in ({"optimizer_step_seconds": []},
                            {"optimizer_step_seconds": [float("nan")]},
                            {"optimizer_step_seconds": [1], "setup_seconds": -1}):
            with self.subTest(calibration=calibration), self.assertRaises(ValueError):
                estimate(calibration, **defaults)


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.events = self.root / "events.jsonl"
        self.manifest.write_text('{"status":"running"}', encoding="utf-8")

    def test_partial_last_event_keeps_last_committed_record(self):
        self.events.write_bytes(b'{"step":1}\n{"step":2}\n{"step":')
        self.assertEqual(read_status(self.manifest, self.events)["latest_event"], {"step": 2})

    def test_long_unicode_event_and_trailing_blank_lines(self):
        event = {"text": "\u03c1" * 20000, "step": 3}
        self.events.write_text(json.dumps(event, ensure_ascii=False) + "\n\n", encoding="utf-8")
        self.assertEqual(read_status(self.manifest, self.events)["latest_event"], event)

    def test_missing_events_and_terminal_poll_do_not_sleep_or_write(self):
        self.manifest.write_text('{"status":"cancelled"}', encoding="utf-8")
        before = self.manifest.read_bytes()
        with patch("tools.watch_run.time.sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(watch(self.manifest, self.events, interval=1, max_polls=5), "cancelled")
        sleep.assert_not_called()
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertFalse(self.events.exists())

    def test_poll_bound_and_record_corruption(self):
        with patch("tools.watch_run.time.sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(watch(self.manifest, self.events, interval=1, max_polls=2), "running")
        sleep.assert_called_once_with(1)
        self.events.write_bytes(b'{"step":1}\nbad completed event\n')
        with self.assertRaises(ValueError):
            read_status(self.manifest, self.events)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "nested").mkdir()
        (self.root / "nested" / "checkpoint.pt").write_bytes(b"checkpoint")
        (self.root / "summary.json").write_bytes(b"{}")
        self.inventory = self.root / "transfer_inventory.json"
        create(self.root, self.inventory)

    def test_nested_artifacts_and_inventory_self_exclusion(self):
        result = verify(self.root, self.inventory, strict=True)
        self.assertTrue(result["verified"])
        self.assertEqual(result["files"], 2)
        self.assertEqual(result["total_bytes"], 12)

    def test_same_size_corruption_missing_and_extra_files(self):
        (self.root / "summary.json").write_bytes(b"[]")
        (self.root / "nested" / "checkpoint.pt").unlink()
        (self.root / "unexpected.json").write_bytes(b"{}")
        result = verify(self.root, self.inventory, strict=True)
        self.assertFalse(result["verified"])
        self.assertEqual(len(result["errors"]), 3)

    def test_reject_escape_duplicate_and_bad_inventory_total(self):
        original = json.loads(self.inventory.read_text(encoding="utf-8"))
        for name in ("../outside", "C:/outside", "nested\\checkpoint.pt", "/outside"):
            broken = json.loads(json.dumps(original))
            broken["files"][0]["path"] = name
            self.inventory.write_text(json.dumps(broken), encoding="utf-8")
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify(self.root, self.inventory)
        for broken in (original | {"files": original["files"] * 2},
                       original | {"total_bytes": 0}, [],
                       original | {"files": [None]}):
            self.inventory.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(self.root, self.inventory)


if __name__ == "__main__":
    unittest.main()
