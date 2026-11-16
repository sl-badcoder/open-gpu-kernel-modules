#!/usr/bin/env python3
"""Exercise telemetry with a fake proc/sys tree and a real child process."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import monitor_uvm as monitor


class MonitorTest(unittest.TestCase):
    def test_counter_reset_and_gauge(self):
        before = {"monotonic_s": 10, "metrics": {
            "vab/GPU/0/prefetch_refault_returns": 100,
            "vab/GPU/0/usage_percent": 90,
            "module/uvm_cpu_preferred_prefetched_bytes": 4096}}
        after = {"monotonic_s": 12, "metrics": {
            "vab/GPU/0/prefetch_refault_returns": 110,
            "vab/GPU/0/usage_percent": 50,
            "module/uvm_cpu_preferred_prefetched_bytes": 0}}
        rows = {r["metric"]: r for r in monitor.differences(before, after)}
        self.assertEqual(rows["vab/GPU/0/prefetch_refault_returns"]["per_second"], 5)
        self.assertIsNone(rows["vab/GPU/0/usage_percent"]["per_second"])
        reset = rows["module/uvm_cpu_preferred_prefetched_bytes"]
        self.assertTrue(reset["reset_detected"])
        self.assertIsNone(reset["change"])

    def test_split_byte_counters_have_rates(self):
        for part in ("trigger", "neighbor"):
            key = f"module/uvm_cpu_preferred_{part}_prefetched_bytes"
            row = monitor.differences({"monotonic_s": 10, "metrics": {key: 4096}},
                                      {"monotonic_s": 12, "metrics": {key: 12288}})[0]
            self.assertEqual(row["kind"], "counter")
            self.assertEqual(row["change"], 8192)
            self.assertEqual(row["per_second"], 4096)

    def test_replacement_counters_and_gauges(self):
        counters = ("turnover_trigger_population_bytes", "turnover_repopulated_bytes",
                    "turnover_short_quarter_bytes", "replacement_selection_ns",
                    "replacement_freed_roots", "turnover_epoch_pauses",
                    "background_cached_root_releases", "background_allocated_root_releases")
        gauges = ("turnover_resident_bytes", "replacement_pending_roots",
                  "turnover_request_stride", "turnover_last_population_bytes",
                  "turnover_epoch_roots_used", "turnover_epoch_roots_limit",
                  "replacement_cached_free_bytes", "replacement_usable_free_bytes")
        for name in counters + gauges:
            key = "vab/GPU/0/" + name
            row = monitor.differences({"monotonic_s": 0, "metrics": {key: 4}},
                                      {"monotonic_s": 2, "metrics": {key: 8}})[0]
            self.assertEqual(row["per_second"], 2 if name in counters else None)

    def test_run_collects_changes_and_preserves_child_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / "sys/module/nvidia_uvm/parameters"
            params.mkdir(parents=True)
            counter = params / "uvm_cpu_preferred_prefetched_bytes"
            counter.write_text("100\n")
            vab = root / "proc/driver/nvidia-uvm/gpus/GPU-test/0/vab"
            vab.parent.mkdir(parents=True)
            vab.write_text("linear_base 0x400000\nusage_percent 90\n"
                           "prefetch_refault_returns 10\nprefetch_refault_short_returns 5\n"
                           "turnover_trigger_population_bytes 100\nturnover_short_quarter_bytes 10\n"
                           "access_bits 00000001 00000002\n")
            def fake_command(args, timeout=5):
                query = next((a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None)
                # An optional unsupported field must not prevent basic telemetry.
                if query == "power.draw":
                    return {"returncode": 1, "stdout": "", "stderr": "unsupported"}
                fields = query.split(",") if query else []
                output = ", ".join("GPU-test" if f == "uuid" else "42" for f in fields)
                return {"returncode": 0, "stdout": output + "\n", "stderr": ""}
            out = root / "results"
            program = ("from pathlib import Path; import time; "
                       f"Path({str(vab)!r}).write_text('prefetch_refault_returns 30\\nprefetch_refault_short_returns 15\\n"
                       "turnover_trigger_population_bytes 500\\nturnover_short_quarter_bytes 210\\n'); "
                       "time.sleep(0.35); "
                       f"Path({str(counter)!r}).write_text('500'); "
                       f"Path({str(vab)!r}).unlink(); "
                       "print('benchmark output'); raise SystemExit(7)")
            with patch.object(monitor, "ROOT", root), patch.object(monitor, "command", fake_command), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = monitor.main(["--output-dir", str(out), "--interval", "0.1",
                                     "--no-pcie", "--", sys.executable, "-c", program])
            self.assertEqual(code, 7)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["exit_code"], 7)
            change = next(r for r in summary["changes"] if r["metric"].endswith("prefetched_bytes"))
            self.assertEqual(change["change"], 400)
            # Per-GPU proc files can disappear at CUDA teardown. Their last
            # available readings must survive in the observed summary.
            self.assertTrue(summary["vab_seen"])
            observed = next(r for r in summary["observed_changes"]
                            if r["metric"].endswith("/prefetch_refault_returns"))
            self.assertEqual(observed["change"], 20)
            self.assertGreater(observed["observation_span_s"], 0)
            self.assertEqual(summary["derived"]["vab/GPU-test/0/short_return_fraction"], 0.5)
            self.assertEqual(summary["derived"]["vab/GPU-test/0/turnover_short_quarter_fraction"], 0.5)
            self.assertIn("benchmark output", (out / "benchmark.log").read_text())
            before = json.loads((out / "before.json").read_text())
            self.assertEqual(before["metrics"]["vab/GPU-test/0/linear_base"], 0x400000)
            self.assertNotIn("gpu/GPU-test/power.draw", before["metrics"])
            self.assertIn("access_bits", before["raw"][str(vab.relative_to(root))])
            with (out / "samples.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertTrue(any(r["counter_per_second"] for r in rows))
            self.assertGreaterEqual(len((out / "samples.jsonl").read_text().splitlines()), 3)

    def test_missing_metrics_do_not_abort(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(monitor, "ROOT", Path(directory)), \
             patch.object(monitor, "command", return_value={"returncode": None, "error": "missing"}):
            item = monitor.snapshot("0")
            self.assertIn("vab", item["errors"])
            self.assertIn("module", item["errors"])
            self.assertIn("nvidia-smi", item["errors"])

    def test_vab_discovered_after_context_appears(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(monitor, "ROOT", Path(directory)), \
             patch.object(monitor, "command", return_value={"returncode": None, "error": "missing"}):
            first = monitor.snapshot("0")
            self.assertIn("vab", first["errors"])
            vab = Path(directory) / "proc/driver/nvidia-uvm/gpus/GPU-new/0/vab"
            vab.parent.mkdir(parents=True)
            vab.write_text("samples 123\n")
            (vab.parent.parent.parent / "1").symlink_to(vab.parent, target_is_directory=True)
            second = monitor.snapshot("0")
            self.assertNotIn("vab", second["errors"])
            self.assertEqual(second["metrics"]["vab/GPU-new/0/samples"], 123)
            self.assertEqual(sum(k.startswith("vab/") for k in second["metrics"]), 1)

    def test_missing_executable_still_saves_after_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(monitor, "ROOT", Path(directory)), \
             patch.object(monitor, "command", return_value={"returncode": None, "error": "missing"}), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out = Path(directory) / "run"
            result = monitor.main(["--output-dir", str(out), "--no-pcie", "--",
                                   str(Path(directory) / "nonexistent")])
            self.assertEqual(result, 127)
            self.assertTrue((out / "after.json").exists())
            self.assertEqual(json.loads((out / "summary.json").read_text())["exit_code"], 127)

    def test_interruption_preserves_checkpoint_and_final_snapshot(self):
        original_snapshot = monitor.snapshot
        count = 0
        def interrupted_snapshot(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt
            return original_snapshot(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(monitor, "ROOT", Path(directory)), \
             patch.object(monitor, "command", return_value={"returncode": None, "error": "missing"}), \
             patch.object(monitor, "snapshot", interrupted_snapshot), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out = Path(directory) / "run"
            code = monitor.main(["--output-dir", str(out), "--interval", "0.1", "--no-pcie", "--",
                                 sys.executable, "-c", "import time; time.sleep(10)"])
            self.assertEqual(code, 130)
            self.assertTrue((out / "after.json").exists())
            self.assertTrue((out / "last_sample.json").exists())
            status = json.loads((out / "run_status.json").read_text())
            self.assertEqual(status["state"], "finished")
            self.assertEqual(status["exit_code"], 130)


if __name__ == "__main__":
    unittest.main()
