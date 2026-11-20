#!/usr/bin/env python3
"""Plan or run staged UVM sensitivity experiments from the vLLM fork directory."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def configurations(phase, batch_count=16, reserve=64, pressure=85, extra=0):
    reference = dict(replacement=1, extra=extra, batch_count=batch_count,
                     reserve_mb=reserve, pressure=pressure)
    if phase == "admission":
        return [dict(reference, label="legacy", replacement=0, extra=0),
                dict(reference, label="bounded", extra=0),
                dict(reference, label="bounded-extra2", extra=2)]
    axis = {"refill": ("batch_count", [4, 8, 16]),
            "reserve": ("reserve_mb", [32, 64, 128]),
            "pressure": ("pressure", [80, 85, 90])}
    if phase in axis:
        key, values = axis[phase]
        return [dict(reference, label=f"{key}-{v}", **{key: v}) for v in values]
    return [dict(reference, label="validation")]


def parameters(config):
    return dict(
        uvm_cpu_preferred_blocks_enable=1,
        uvm_cpu_preferred_block_promotion_enable=0,
        uvm_cpu_preferred_prefetch_chunk_mb=64,
        uvm_global_oversubscription=1,
        uvm_perf_access_counter_migration_enable=1,
        uvm_perf_access_counter_threshold=1024,
        uvm_perf_access_counter_admission_enable=1,
        uvm_perf_access_counter_admission_pressure_percent=config["pressure"],
        uvm_perf_access_counter_admission_observations=2,
        uvm_perf_access_counter_admission_eviction_enable=int(config["extra"] > 0),
        uvm_perf_access_counter_admission_eviction_extra=config["extra"],
        uvm_perf_access_counter_admission_eviction_mb=256,
        uvm_perf_access_counter_admission_distance_stats_enable=1,
        uvm_perf_prefetch_replacement_enable=config["replacement"],
        uvm_perf_eager_eviction_enable=1,
        uvm_perf_eager_eviction_high_watermark=99,
        uvm_perf_eager_eviction_low_watermark=90,
        uvm_perf_eager_eviction_interval_ms=25,
        uvm_perf_eager_eviction_batch_count=config["batch_count"],
        uvm_perf_vab_mode=3,
        uvm_perf_vab_eviction_linear_base=0,
        uvm_perf_speculative_prefetch_backoff_enable=0,
        uvm_perf_speculative_prefetch_recovery_enable=0,
        uvm_perf_speculative_prefetch_sparse_enable=0,
        uvm_perf_speculative_prefetch_limit_mb=32,
        uvm_perf_speculative_prefetch_reserve_mb=config["reserve_mb"],
        uvm_perf_prefetch_refault_enable=0)


def plan(configs, batches, rounds, seed):
    rng = random.Random(seed)
    jobs = []
    for repeat in range(1, rounds + 1):
        group = list(itertools.product(batches, configs))
        rng.shuffle(group)
        jobs.extend(dict(repeat=repeat, batch=batch, config=config) for batch, config in group)
    return jobs


def normalize(value):
    return {"Y": "1", "N": "0"}.get(str(value).strip(), str(value).strip())


def reload_module(module, expected):
    if Path("/sys/module/nvidia_uvm").exists():
        subprocess.run(["sudo", "rmmod", "nvidia_uvm"], check=True)
    subprocess.run(["sudo", "insmod", str(module)] +
                   [f"{key}={value}" for key, value in expected.items()], check=True)
    actual = {key: (Path("/sys/module/nvidia_uvm/parameters") / key).read_text().strip()
              for key in expected}
    mismatch = {key: actual[key] for key, value in expected.items()
                if normalize(actual[key]) != str(value)}
    if mismatch:
        raise RuntimeError(f"Loaded parameter mismatch: {mismatch}")
    return actual


def read_run(directory):
    manifest = json.loads((directory / "sensitivity.json").read_text())
    row = dict(run=str(directory), repeat=manifest["repeat"], batch=manifest["batch"],
               label=manifest["config"]["label"], **{
                   k: v for k, v in manifest["config"].items() if k != "label"})
    row["status"] = "incomplete"
    if manifest.get("reload_error"):
        row.update(status="reload_failed", error=manifest["reload_error"])
        return row
    row["phase"] = manifest.get("phase", "unknown")
    if not (directory / "summary.json").exists() or not (directory / "benchmark.json").exists():
        return row
    summary = json.loads((directory / "summary.json").read_text())
    benchmark = json.loads((directory / "benchmark.json").read_text())
    rounds = benchmark.get("rounds", [])
    if manifest.get("monitor_exit_code", 0) != 0 or summary["exit_code"] != 0 or len(rounds) < 3:
        row["status"] = "failed"
        return row
    warm = rounds[1:]
    output = sum(r["seconds"] * r["output_tokens_per_second"] for r in rounds)
    row.update(status="complete", cold_tps=rounds[0]["output_tokens_per_second"],
               warm_tps=sum(r["seconds"] * r["output_tokens_per_second"] for r in warm) /
                        sum(r["seconds"] for r in warm),
               output_tokens=round(output), span_seconds=summary["snapshot_span_seconds"])
    changes = {r["metric"].split("/")[-1]: r["change"]
               for r in summary.get("observed_changes", summary["changes"])}
    for column, name in (
        ("population_bytes", "uvm_cpu_preferred_prefetched_bytes"),
        ("neighbor_bytes", "uvm_cpu_preferred_neighbor_prefetched_bytes"),
        ("root_releases", "background_roots_freed"),
        ("capacity_stops", "uvm_cpu_preferred_prefetch_capacity_stops"),
        ("replacement_requests", "replacement_requests")):
        row[column] = changes.get(name)
    if row["population_bytes"] is not None:
        row["population_gib"] = row["population_bytes"] / 2**30
        row["population_mib_per_output_token"] = row["population_bytes"] / 2**20 / output
    if row["root_releases"] is not None:
        row["root_releases_per_1k_output_tokens"] = row["root_releases"] * 1000 / output
    hist = summary.get("return_distance_histogram")
    if hist:
        total = sum(r["population_bytes"] for r in hist)
        for entry in hist:
            row[f"return_{entry['bucket']}_byte_fraction"] = entry["population_bytes"] / total if total else None
    return row


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows):
    references = dict(admission="bounded", refill="batch_count-16",
                      reserve="reserve_mb-64", pressure="pressure-85")
    groups = {}
    for row in rows:
        if row["status"] == "complete":
            groups.setdefault((row.get("phase"), row["batch"], row["repeat"]), []).append(row)
    pairs = []
    for (phase, batch, repeat), runs in groups.items():
        reference = next((r for r in runs if r["label"] == references.get(phase)), None)
        if reference is None:
            continue
        for candidate in runs:
            if candidate is reference:
                continue
            pair = dict(phase=phase, batch=batch, repeat=repeat, reference=reference["label"],
                        candidate=candidate["label"])
            for metric in ("warm_tps", "population_mib_per_output_token", "root_releases_per_1k_output_tokens"):
                a, b = candidate.get(metric), reference.get(metric)
                pair[metric + "_ratio"] = a / b if a is not None and b else None
            pairs.append(pair)
    return pairs


def summarize(root):
    rows = [read_run(p.parent) for p in sorted(root.glob("**/sensitivity.json"))]
    if not rows:
        raise ValueError(f"No sensitivity run manifests under {root}")
    write_csv(root / "runs.csv", rows)
    groups = {}
    for row in rows:
        if row["status"] == "complete":
            key = tuple(row[k] for k in ("batch", "label", "replacement", "extra", "batch_count", "reserve_mb", "pressure"))
            groups.setdefault(key, []).append(row)
    aggregate = []
    for key, runs in sorted(groups.items()):
        item = dict(zip(("batch", "label", "replacement", "extra", "batch_count", "reserve_mb", "pressure"), key))
        item["independent_runs"] = len(runs)
        for metric in ("warm_tps", "cold_tps", "population_mib_per_output_token", "root_releases_per_1k_output_tokens"):
            values = [r[metric] for r in runs if r.get(metric) is not None]
            if values:
                item[f"{metric}_median"] = statistics.median(values)
                item[f"{metric}_min"] = min(values)
                item[f"{metric}_max"] = max(values)
        aggregate.append(item)
    write_csv(root / "aggregate.csv", aggregate)
    write_csv(root / "paired.csv", paired_rows(rows))
    print(f"Saved runs.csv, aggregate.csv and paired.csv under {root}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("admission", "refill", "reserve", "pressure", "validation"), default="admission")
    parser.add_argument("--batches", type=int, nargs="+", default=[32])
    parser.add_argument("--rounds", type=int, default=3, help="Independent module reloads; each benchmark itself has 3 passes")
    parser.add_argument("--batch-count", type=int, default=16)
    parser.add_argument("--reserve-mb", type=int, default=64)
    parser.add_argument("--pressure", type=int, default=85)
    parser.add_argument("--extra", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--pcie", action="store_true", help="Also run the offline dmon sampler")
    parser.add_argument("--module", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/uvm-sensitivity"))
    parser.add_argument("--execute", action="store_true", help="Reload the module and run the displayed plan")
    parser.add_argument("--summarize", type=Path, help="Rebuild CSV summaries for an existing suite, without running anything")
    args = parser.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0
    if (args.rounds < 1 or any(b < 1 or b > 32 for b in args.batches) or
        not 1 <= args.batch_count <= 16 or args.reserve_mb < 0 or
        not 1 <= args.pressure <= 100 or args.interval < 0.1):
        parser.error("Invalid batch, round, pressure, reserve, or interval")
    jobs = plan(configurations(args.phase, args.batch_count, args.reserve_mb, args.pressure, args.extra),
                args.batches, args.rounds, args.seed)
    print(json.dumps(jobs, indent=2))
    print(f"{len(jobs)} independent runs; {len(jobs)*3} benchmark passes. Module reload before every run.")
    if not args.execute:
        print("Plan only. Add --execute --module /path/to/nvidia-uvm.ko to run.")
        return 0
    if not args.module or not args.module.is_file():
        parser.error("--execute requires an existing --module file")
    if not Path(".venv/bin/python").is_file() or not Path("benchmarks/benchmark_uvm.py").is_file():
        parser.error("Run from the vLLM fork directory")
    module = args.module.resolve()
    digest = hashlib.sha256()
    with module.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = args.output_root / f"{args.phase}-{stamp}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "plan.json").write_text(json.dumps(dict(jobs=jobs, module=str(module), module_sha256=digest.hexdigest()), indent=2))
    try:
        for index, job in enumerate(jobs, 1):
            directory = root / f"{index:03d}-r{job['repeat']}-b{job['batch']}-{job['config']['label']}"
            expected = parameters(job["config"])
            try:
                actual = reload_module(module, expected)
            except (OSError, RuntimeError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
                directory.mkdir(exist_ok=False)
                (directory / "sensitivity.json").write_text(json.dumps(dict(job, phase=args.phase,
                    expected=expected, reload_error=repr(error)), indent=2))
                summarize(root)
                print(f"Module reload failed; stopped suite: {error}", file=sys.stderr)
                return 130 if isinstance(error, KeyboardInterrupt) else 1
            command = [sys.executable, str(HERE / "monitor_uvm.py"), "--gpu", args.gpu,
                       "--interval", str(args.interval), "--output-dir", str(directory)]
            if not args.pcie:
                command.append("--no-pcie")
            command += ["--", ".venv/bin/python", "benchmarks/benchmark_uvm.py",
                        "--model", "Qwen/Qwen2.5-3B-Instruct", "--allocator", "managed", "--kv-cache-gib", "12",
                        "--require-oversubscription", "--num-prompts", "32", "--batch-size", str(job["batch"]),
                        "--input-tokens", "10240", "--output-tokens", "256", "--repeats", "3",
                        "--output", str(directory / "benchmark.json")]
            # The monitor creates its output directory exclusively.
            exit_code = 130
            try:
                result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu))
                exit_code = result.returncode
            finally:
                directory.mkdir(exist_ok=True)
                (directory / "sensitivity.json").write_text(json.dumps(dict(job, phase=args.phase, expected=expected, actual=actual,
                                                           monitor_exit_code=exit_code), indent=2))
                summarize(root)
            if exit_code:
                return exit_code
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
