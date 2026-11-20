#!/usr/bin/env python3
"""Validate matched plans and summaries without reloading a GPU driver."""
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import run_vllm_sensitivity as sweep


class SensitivityTest(unittest.TestCase):
    def test_paired_comparisons_match_repeat(self):
        rows = []
        for repeat, reference in ((1, 100), (2, 200)):
            for label, rate in (("bounded", reference), ("bounded-extra2", reference * 1.2)):
                rows.append(dict(status="complete", phase="admission", repeat=repeat, batch=32,
                                 label=label, warm_tps=rate))
        pairs = sweep.paired_rows(rows)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(p["warm_tps_ratio"] == 1.2 for p in pairs))

    def test_execute_reloads_every_run_and_preserves_unique_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv/bin").mkdir(parents=True)
            (root / ".venv/bin/python").touch()
            (root / "benchmarks").mkdir()
            (root / "benchmarks/benchmark_uvm.py").touch()
            module = root / "nvidia-uvm.ko"
            module.write_bytes(b"test module")
            created = []

            def fake_monitor(command, **kwargs):
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-test")
                directory = Path(command[command.index("--output-dir") + 1])
                directory.mkdir(exist_ok=False)
                created.append(directory.resolve())
                return type("Result", (), {"returncode": 0})()

            original = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sweep, "reload_module", return_value={}) as reload, \
                     patch.object(sweep.subprocess, "run", side_effect=fake_monitor):
                    self.assertEqual(sweep.main(["--phase", "validation", "--rounds", "1",
                        "--batches", "8", "24", "--gpu", "GPU-test", "--execute", "--module", str(module)]), 0)
                    self.assertEqual(reload.call_count, 2)
            finally:
                os.chdir(original)
            self.assertEqual(len(set(created)), 2)
            self.assertTrue(all((d / "sensitivity.json").exists() for d in created))

    def test_axes_and_independent_rounds(self):
        configs = sweep.configurations("refill")
        self.assertEqual([c["batch_count"] for c in configs], [4, 8, 16])
        self.assertEqual({c["reserve_mb"] for c in configs}, {64})
        jobs = sweep.plan(configs, [8, 24, 32], 3, 42)
        self.assertEqual(len(jobs), 27)
        self.assertEqual(jobs, sweep.plan(configs, [8, 24, 32], 3, 42))
        for repeat in (1, 2, 3):
            subset = [j for j in jobs if j["repeat"] == repeat]
            self.assertEqual(len({(j["batch"], j["config"]["label"]) for j in subset}), 9)

    def test_registered_parameters_and_matched_comparisons(self):
        source = "\n".join(p.read_text() for p in (sweep.HERE.parents[1] / "kernel-open/nvidia-uvm").glob("*.c"))
        registered = set(re.findall(r"module_param(?:_cb|_named)?\(\s*(\w+)", source))
        old, bounded, extra = map(sweep.parameters, sweep.configurations("admission"))
        self.assertFalse(set(bounded) - registered)
        self.assertEqual([k for k in old if old[k] != bounded[k]], ["uvm_perf_prefetch_replacement_enable"])
        self.assertEqual(set(k for k in extra if extra[k] != bounded[k]), {
            "uvm_perf_access_counter_admission_eviction_enable", "uvm_perf_access_counter_admission_eviction_extra"})

    def test_plan_does_not_execute(self):
        with patch.object(sweep.subprocess, "run") as run:
            self.assertEqual(sweep.main(["--rounds", "1"]), 0)
            run.assert_not_called()

    def test_success_failure_and_missing_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "run"
            d.mkdir()
            manifest = dict(repeat=1, batch=32, config=sweep.configurations("validation")[0])
            (d / "sensitivity.json").write_text(json.dumps(manifest))
            self.assertEqual(sweep.read_run(d)["status"], "incomplete")
            (d / "sensitivity.json").write_text(json.dumps(dict(manifest, reload_error="module busy")))
            self.assertEqual(sweep.read_run(d)["status"], "reload_failed")
            (d / "sensitivity.json").write_text(json.dumps(manifest))
            benchmark = {"rounds": [dict(seconds=10, output_tokens_per_second=10),
                                      dict(seconds=10, output_tokens_per_second=20),
                                      dict(seconds=20, output_tokens_per_second=10)]}
            (d / "benchmark.json").write_text(json.dumps(benchmark))
            summary = dict(exit_code=0, snapshot_span_seconds=50, changes=[], observed_changes=[
                dict(metric="module/uvm_cpu_preferred_prefetched_bytes", change=500*2**20),
                dict(metric="vab/GPU/0/background_roots_freed", change=50)])
            (d / "summary.json").write_text(json.dumps(summary))
            row = sweep.read_run(d)
            self.assertEqual(row["status"], "complete")
            self.assertAlmostEqual(row["warm_tps"], 400/30)
            self.assertEqual(row["population_mib_per_output_token"], 1)
            self.assertEqual(row["root_releases_per_1k_output_tokens"], 100)
            self.assertIsNone(row["capacity_stops"])
            sweep.summarize(root)
            self.assertTrue((root / "aggregate.csv").is_file())
            summary["exit_code"] = 130
            (d / "summary.json").write_text(json.dumps(summary))
            self.assertEqual(sweep.read_run(d)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
