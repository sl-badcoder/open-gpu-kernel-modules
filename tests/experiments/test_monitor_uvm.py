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
    def test_placement_diagnostics_common_interval_and_units(self):
        names = ["counter_service", "counter_block", "population_submit", "population_wait",
                 "eviction_select", "eviction_submit", "eviction_drain"]
        before_metrics = {"module/uvm_perf_placement_diagnostics_enable": "Y"}
        after_metrics = {"module/uvm_perf_placement_diagnostics_enable": "Y"}
        for index, name in enumerate(names, 1):
            before_metrics[f"module/uvm_perf_diag_{name}_calls"] = 10
            after_metrics[f"module/uvm_perf_diag_{name}_calls"] = 10 + index
            before_metrics[f"module/uvm_perf_diag_{name}_ns"] = 1000
            after_metrics[f"module/uvm_perf_diag_{name}_ns"] = 1000 + index * 1000
        for name, value in (("empty", 2), ("nonempty", 6), ("failed", 1)):
            before_metrics[f"module/uvm_perf_diag_admitted_{name}"] = 4
            after_metrics[f"module/uvm_perf_diag_admitted_{name}"] = 4 + value
        rows = monitor.differences({"monotonic_s": 10, "metrics": before_metrics},
                                   {"monotonic_s": 12, "metrics": after_metrics})
        counter_block_rows = {row["metric"]: row for row in rows
                              if "counter_block" in row["metric"]}
        self.assertEqual(counter_block_rows["module/uvm_perf_diag_counter_block_calls"]["kind"],
                         "counter")
        result = monitor.placement_diagnostics(rows, before_metrics, after_metrics)
        self.assertEqual(result["status"], "enabled")
        self.assertEqual(result["stages"]["counter_service"]["calls"], 1)
        self.assertEqual(result["stages"]["counter_service"]["total_seconds"], 1e-6)
        self.assertEqual(result["stages"]["counter_service"]["mean_us"], 1.0)
        self.assertEqual(result["admitted"]["empty"], 2)
        self.assertEqual(result["admitted"]["nonempty"], 6)
        self.assertEqual(result["admitted"]["failed"], 1)
        self.assertEqual(result["admitted"]["empty_fraction"], 0.25)

    def test_placement_diagnostics_disabled_missing_and_reset(self):
        key = "module/uvm_perf_diag_counter_service_calls"
        ns = "module/uvm_perf_diag_counter_service_ns"
        before = {"module/uvm_perf_placement_diagnostics_enable": "N", key: 10, ns: 100}
        after = {"module/uvm_perf_placement_diagnostics_enable": "N", key: 11, ns: 200}
        result = monitor.placement_diagnostics(
            monitor.differences({"monotonic_s": 1, "metrics": before},
                                 {"monotonic_s": 2, "metrics": after}), before, after)
        self.assertEqual(result["status"], "disabled")
        self.assertEqual(result["stages"]["counter_service"]["mean_us"], 0.1)
        self.assertIsNone(result["admitted"]["empty_fraction"])

        missing = monitor.placement_diagnostics([], {}, {})
        self.assertEqual(missing["status"], "unsupported")
        self.assertIsNone(missing["stages"]["counter_service"]["calls"])
        self.assertIsNone(missing["admitted"]["empty_fraction"])

        reset_before = {"module/uvm_perf_placement_diagnostics_enable": 1,
                        "module/uvm_perf_diag_admitted_empty": 9,
                        "module/uvm_perf_diag_admitted_nonempty": 9}
        reset_after = {"module/uvm_perf_placement_diagnostics_enable": 1,
                       "module/uvm_perf_diag_admitted_empty": 1,
                       "module/uvm_perf_diag_admitted_nonempty": 1}
        reset = monitor.placement_diagnostics(
            monitor.differences({"monotonic_s": 1, "metrics": reset_before},
                                 {"monotonic_s": 2, "metrics": reset_after}),
            reset_before, reset_after)
        self.assertIsNone(reset["admitted"]["empty"])
        self.assertIsNone(reset["admitted"]["nonempty"])
        self.assertIsNone(reset["admitted"]["empty_fraction"])

        changed = monitor.placement_diagnostics(
            [], {"module/uvm_perf_placement_diagnostics_enable": "N"},
            {"module/uvm_perf_placement_diagnostics_enable": "Y"})
        self.assertEqual(changed["status"], "changed")
        unknown = monitor.placement_diagnostics(
            [], {"module/uvm_perf_placement_diagnostics_enable": "maybe"},
            {"module/uvm_perf_placement_diagnostics_enable": "maybe"})
        self.assertEqual(unknown["status"], "unknown")

    def test_trigger_attribution(self):
        delta = {"module/uvm_cpu_preferred_trigger_indicated_bytes": 256*1024,
                 "module/uvm_cpu_preferred_trigger_extra_bytes": 1792*1024}
        self.assertEqual(monitor.trigger_extra_fraction(delta), 0.875)
        self.assertIsNone(monitor.trigger_extra_fraction({}))
        self.assertIsNone(monitor.trigger_extra_fraction({k: 0 for k in delta}))
        self.assertIsNone(monitor.trigger_extra_fraction({k: None for k in delta}))
        rows = monitor.differences({"monotonic_s": 10, "metrics": {k: 0 for k in delta}},
                                   {"monotonic_s": 12, "metrics": delta})
        self.assertTrue(all(r["kind"] == "counter" and r["per_second"] == r["change"]/2 for r in rows))

    def test_return_distance_counters_and_histogram(self):
        before = {"monotonic_s": 10, "metrics": {}}
        after = {"monotonic_s": 12, "metrics": {}}
        for i, bucket in enumerate(monitor.RETURN_DISTANCE_BUCKETS):
            for metric in ("probations", "grants", "populations", "population_bytes"):
                key = f"module/uvm_cpu_preferred_return_{bucket}_{metric}"
                before["metrics"][key] = 100
                after["metrics"][key] = 100 + i * 4096
        rows = monitor.differences(before, after)
        self.assertTrue(all(row["kind"] == "counter" for row in rows))
        self.assertTrue(all(row["per_second"] == row["change"] / 2 for row in rows))
        histogram = monitor.return_distance_histogram({r["metric"]: r["change"] for r in rows})
        self.assertEqual(histogram[4]["population_bytes"], 16384)
        self.assertEqual(histogram[4]["population_byte_fraction"], 0.4)
        self.assertIsNone(monitor.return_distance_histogram({}))
        self.assertTrue(all(row["population_byte_fraction"] is None for row in
                            monitor.return_distance_histogram({r["metric"]: 0 for r in rows})))

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

    def test_coverage_admission_counter_rates(self):
        for suffix in ("broad", "sparse", "deferred"):
            key = f"module/uvm_cpu_preferred_admission_coverage_{suffix}"
            row = monitor.differences({"monotonic_s": 10, "metrics": {key: 3}},
                                      {"monotonic_s": 12, "metrics": {key: 9}})[0]
            self.assertEqual(row["kind"], "counter")
            self.assertEqual(row["per_second"], 3)

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
                       f"Path({str(vab)!r}).write_text('prefetch_refault_returns 30\\nprefetch_refault_short_returns 15\\n'); "
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
