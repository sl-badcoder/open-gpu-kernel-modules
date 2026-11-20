#!/usr/bin/env python3
"""Run a command with read-only UVM/GPU telemetry. Python standard library only."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time


ROOT = Path("/")
GPU_FIELDS = ("uuid", "name", "memory.total", "memory.used", "memory.free",
              "utilization.gpu", "utilization.memory", "temperature.gpu",
              "power.draw", "clocks.sm", "clocks.mem",
              "pcie.link.gen.current", "pcie.link.width.current")
# Only known cumulative counters receive rates. Other numeric fields are gauges
# or configuration, whose signed changes must not be called event counts.
COUNTERS = set("""samples dump_errors pm_skips rising_bits falling_bits unchanged_samples
selected_hot selected_cold selected_unknown background_roots_freed background_failures
selection_changes selection_fallbacks background_hot_skips background_unknown_skips
prefetch_admissions prefetch_denials prefetch_backoffs prefetch_quick_returns
prefetch_capacity_stops prefetch_pressure_pauses prefetch_allocation_failures
prefetch_prefetched_blocks prefetch_ratio_evaluations prefetch_discarded_blocks
prefetch_discarded_returns prefetch_dropped_records prefetch_dropped_returns
prefetch_refault_demand_bytes prefetch_refault_evictions prefetch_refault_returns
prefetch_refault_short_returns prefetch_refault_replacements prefetch_refault_plans
prefetch_refault_paused prefetch_refault_growths prefetch_refault_reductions
prefetch_bad_batches prefetch_good_batches prefetch_holds prefetch_pattern_events
prefetch_pattern_confirmations prefetch_pattern_replacements prefetch_pattern_predictions
prefetch_pattern_learned_predictions prefetch_pattern_expirations prefetch_recoveries
prefetch_probes uvm_cpu_preferred_block_promotions uvm_cpu_preferred_range_prefetches
uvm_cpu_preferred_prefetch_retries uvm_cpu_preferred_prefetched_bytes
uvm_cpu_preferred_trigger_prefetched_bytes uvm_cpu_preferred_neighbor_prefetched_bytes
uvm_cpu_preferred_trigger_indicated_bytes uvm_cpu_preferred_trigger_extra_bytes
uvm_cpu_preferred_prefetch_capacity_stops uvm_cpu_preferred_admission_requests
uvm_cpu_preferred_admission_deferred uvm_cpu_preferred_admission_low_pressure
uvm_cpu_preferred_admission_repeated uvm_cpu_preferred_admission_state_failures
uvm_cpu_preferred_admission_eviction_probations uvm_cpu_preferred_admission_eviction_deferred
uvm_cpu_preferred_admission_return_bytes""".split())
COUNTERS.update(("background_cached_root_releases", "background_allocated_root_releases",
                 "replacement_requests", "replacement_freed_roots"))
RETURN_DISTANCE_BUCKETS = ("lt64m", "64m_256m", "256m_1g", "1g_4g", "ge4g")
COUNTERS.update(f"uvm_cpu_preferred_return_{bucket}_{metric}"
                for bucket in RETURN_DISTANCE_BUCKETS
                for metric in ("probations", "grants", "populations", "population_bytes"))


def return_distance_histogram(delta):
    """Actual return bytes and events, kept separate from admission opportunities."""
    rows = []
    for bucket in RETURN_DISTANCE_BUCKETS:
        row = {"bucket": bucket}
        for metric in ("probations", "grants", "populations", "population_bytes"):
            row[metric] = delta.get(f"module/uvm_cpu_preferred_return_{bucket}_{metric}")
        rows.append(row)
    if any(row[metric] is None for row in rows
           for metric in ("probations", "grants", "populations", "population_bytes")):
        return None
    total = sum(row["population_bytes"] for row in rows)
    for row in rows:
        row["population_byte_fraction"] = row["population_bytes"] / total if total else None
    return rows


def trigger_extra_fraction(delta):
    indicated = delta.get("module/uvm_cpu_preferred_trigger_indicated_bytes")
    extra = delta.get("module/uvm_cpu_preferred_trigger_extra_bytes")
    if indicated is None or extra is None or indicated + extra <= 0:
        return None
    return extra / (indicated + extra)


def value(text):
    text = text.strip()
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return text


def command(args, timeout=5):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": str(exc)}


def snapshot(gpu, gpu_fields=GPU_FIELDS):
    started = time.monotonic()
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "monotonic_s": started, "metrics": {}, "raw": {}, "errors": {}}
    metrics = result["metrics"]
    paths = sorted((ROOT / "sys/module/nvidia_uvm/parameters").glob("*"))
    gpu_dir = ROOT / "proc/driver/nvidia-uvm/gpus"
    # scandir reports permission errors which Path.glob can silently suppress.
    # Canonical UUID directories avoid duplicate readings through GPU symlinks.
    try:
        with os.scandir(gpu_dir) as entries:
            parents = sorted(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))
        result["vab_gpu_directories"] = [p.name for p in parents]
        for parent in parents:
            try:
                with os.scandir(parent) as entries:
                    paths += sorted(Path(e.path) / "vab" for e in entries
                                    if e.is_dir(follow_symlinks=False) and e.name.isdigit())
            except OSError as exc:
                result["errors"][str(parent)] = str(exc)
    except OSError as exc:
        result["errors"][str(gpu_dir)] = str(exc)
    paths += [ROOT / "proc/meminfo", ROOT / "proc/vmstat", ROOT / "proc/loadavg",
              ROOT / "proc/stat"]
    for path in paths:
        key = str(path.relative_to(ROOT))
        try:
            text = path.read_text()
        except OSError as exc:
            result["errors"][key] = str(exc)
            continue
        result["raw"][key] = text
        if "parameters/" in key:
            metrics["module/" + path.name] = value(text)
        elif path.name == "vab":
            prefix = "vab/" + "/".join(path.parts[-3:-1]) + "/"
            for line in text.splitlines():
                fields = line.split(None, 1)
                if len(fields) == 2 and fields[0] not in ("enabled_mask", "access_bits"):
                    metrics[prefix + fields[0]] = value(fields[1])
        elif path.name == "stat":
            for line in text.splitlines():
                fields = line.split()
                if fields and fields[0] == "cpu":
                    names = ("user", "nice", "system", "idle", "iowait", "irq",
                             "softirq", "steal", "guest", "guest_nice")
                    for name, ticks in zip(names, fields[1:]):
                        metrics["host/cpu/" + name + "_ticks"] = value(ticks)
        elif path.name in ("meminfo", "vmstat"):
            for line in text.splitlines():
                fields = line.replace(":", "").split()
                if len(fields) >= 2:
                    # meminfo uses kB; preserve the unit explicitly in the key.
                    suffix = "_kib" if len(fields) == 3 and fields[2] == "kB" else ""
                    metrics["host/" + path.name + "/" + fields[0] + suffix] = value(fields[1])
    if not any(k.startswith("vab/") for k in metrics):
        result["errors"]["vab"] = (
            "No readable VAB statistics now; GPU entries may appear only while a CUDA/UVM "
            "context is registered. Discovery/read errors, if any, are recorded separately.")
    if not any(k.startswith("module/") for k in metrics):
        result["errors"]["module"] = "No readable nvidia_uvm module parameters found"
    query = command(["nvidia-smi", "-i", gpu,
                     "--query-gpu=" + ",".join(gpu_fields),
                     "--format=csv,noheader,nounits"])
    result["gpu_query"] = query
    if query["returncode"] == 0:
        for row in csv.reader(query["stdout"].splitlines()):
            for name, cell in zip(gpu_fields, row):
                cell = cell.strip()
                try:
                    number = float(cell)
                except ValueError:
                    number = cell
                metrics["gpu/" + row[0].strip() + "/" + name] = number
    else:
        result["errors"]["nvidia-smi"] = query
    result["collection_seconds"] = time.monotonic() - started
    return result


def differences(before, after):
    elapsed = after["monotonic_s"] - before["monotonic_s"]
    rows = []
    for key in sorted(before["metrics"].keys() & after["metrics"].keys()):
        a, b = before["metrics"][key], after["metrics"][key]
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        counter = key.split("/")[-1] in COUNTERS
        reset = counter and b < a
        delta = None if reset else b - a
        rows.append({"metric": key, "kind": "counter" if counter else "gauge_or_setting",
                     "before": a, "after": b, "change": delta,
                     "per_second": delta / elapsed if counter and not reset and elapsed > 0 else None,
                     "reset_detected": reset})
    return rows


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="Physical nvidia-smi GPU index or UUID")
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds")
    parser.add_argument("--no-pcie", action="store_true", help="Disable nvidia-smi dmon PCIe sampler")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    child_args = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not child_args or args.interval < 0.1:
        parser.error("Provide a command after -- and an interval >= 0.1 seconds")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite another experiment.
    write_json(out / "run_status.json", {"state": "starting", "monitor_pid": os.getpid()})
    metadata = {"command": child_args, "cwd": os.getcwd(), "gpu": args.gpu,
                "interval_seconds": args.interval, "host": platform.uname()._asdict(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "nvidia_smi": command(["nvidia-smi"]),
                "driver_version": command(["cat", "/proc/driver/nvidia/version"]),
                "monitor_source_commit": command(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"]),
                "benchmark_source_commit": command(["git", "rev-parse", "HEAD"])}
    write_json(out / "metadata.json", metadata)
    # Discover supported fields once; one unsupported optional field must not
    # eliminate memory/utilization telemetry on older driver versions.
    fields = []
    for field in GPU_FIELDS:
        probe = command(["nvidia-smi", "-i", args.gpu, "--query-gpu=" + field,
                         "--format=csv,noheader,nounits"])
        if probe["returncode"] == 0:
            fields.append(field)
    if "uuid" not in fields:
        fields = list(GPU_FIELDS)  # Preserve the failure in every snapshot.
    before = snapshot(args.gpu, fields)
    write_json(out / "before.json", before)
    for key, error in before["errors"].items():
        print(f"monitor warning: {key}: {error}", file=sys.stderr)

    dmon = child = copier = None
    exit_code = 1
    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    with (out / "samples.jsonl").open("w") as samples, \
         (out / "samples.csv").open("w", newline="") as table, \
         (out / "benchmark.log").open("w") as log, \
         (out / "pcie_dmon.log").open("w") as pcie:
        writer = csv.writer(table)
        writer.writerow(["timestamp_utc", "elapsed_s", "collection_seconds", "metric", "value",
                         "counter_change", "counter_per_second", "reset_detected"])
        previous = None
        observed = {}
        def record(item):
            nonlocal previous
            # A checkpoint is a measurement during the run, never a fabricated
            # after snapshot. It remains useful if the monitor is killed abruptly.
            write_json(out / "last_sample.json", item)
            samples.write(json.dumps(item) + "\n")
            samples.flush()
            deltas = {r["metric"]: r for r in differences(previous, item)} if previous else {}
            for key, val in sorted(item["metrics"].items()):
                if isinstance(val, (int, float)):
                    reading = {"monotonic_s": item["monotonic_s"], "metrics": {key: val}}
                    if key not in observed:
                        observed[key] = [reading, reading]
                    else:
                        observed[key][1] = reading
                change = deltas.get(key, {})
                is_counter = change.get("kind") == "counter"
                writer.writerow([item["timestamp_utc"], item["monotonic_s"] - before["monotonic_s"],
                                 item["collection_seconds"], key, val,
                                 change.get("change") if is_counter else None,
                                 change.get("per_second") if is_counter else None,
                                 change.get("reset_detected") if is_counter else None])
            table.flush()
            previous = item
        record(before)
        try:
            if not args.no_pcie:
                try:
                    dmon = subprocess.Popen(["nvidia-smi", "dmon", "-i", args.gpu,
                                             "-s", "t", "-d", "1", "-o", "DT"],
                                            stdout=pcie, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                except OSError as exc:
                    pcie.write(str(exc) + "\n")
            print(f"Telemetry: {out.resolve()}", flush=True)
            child = subprocess.Popen(child_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
            write_json(out / "run_status.json", {"state": "running", "monitor_pid": os.getpid(),
                                                "benchmark_pid": child.pid})
            def tee():
                for line in child.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except BrokenPipeError:
                        pass
            copier = threading.Thread(target=tee, daemon=True)
            copier.start()
            while True:
                try:
                    exit_code = child.wait(timeout=args.interval)
                    break
                except subprocess.TimeoutExpired:
                    record(snapshot(args.gpu, fields))
            copier.join(timeout=5)
        except KeyboardInterrupt:
            exit_code = 130
        except OSError as exc:
            log.write(str(exc) + "\n")
            print(str(exc), file=sys.stderr)
            exit_code = 127
        finally:
            # A second Ctrl-C must not interrupt writing the final snapshot.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            write_json(out / "run_status.json", {"state": "finalizing", "monitor_pid": os.getpid(),
                                                "exit_code": exit_code})
            for process in (child, dmon):
                if process is not None and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    except ProcessLookupError:
                        pass
            if copier is not None:
                copier.join(timeout=5)
                if not copier.is_alive():
                    child.stdout.close()
            after = snapshot(args.gpu, fields)
            write_json(out / "after.json", after)
            record(after)
            rows = differences(before, after)
            with (out / "changes.csv").open("w", newline="") as changes:
                columns = ["metric", "kind", "before", "after", "change", "per_second", "reset_detected"]
                delta_writer = csv.DictWriter(changes, fieldnames=columns)
                delta_writer.writeheader()
                delta_writer.writerows(rows)
            observed_rows = []
            for key, (first, last) in sorted(observed.items()):
                row = differences(first, last)[0]
                row["first_seen_elapsed_s"] = first["monotonic_s"] - before["monotonic_s"]
                row["last_seen_elapsed_s"] = last["monotonic_s"] - before["monotonic_s"]
                row["observation_span_s"] = last["monotonic_s"] - first["monotonic_s"]
                observed_rows.append(row)
            with (out / "observed_changes.csv").open("w", newline="") as changes:
                observed_writer = csv.DictWriter(changes, fieldnames=columns + [
                    "first_seen_elapsed_s", "last_seen_elapsed_s", "observation_span_s"])
                observed_writer.writeheader()
                observed_writer.writerows(observed_rows)
            delta = {r["metric"]: r["change"] for r in observed_rows if r["kind"] == "counter"}
            ratios = {}
            ratios["module/trigger_extra_fraction"] = trigger_extra_fraction(delta)
            for key in delta:
                if not key.endswith("/prefetch_refault_returns"):
                    continue
                prefix = key.rsplit("/", 1)[0] + "/"
                for name, numerator, denominator in (
                    ("short_return_fraction", "prefetch_refault_short_returns", "prefetch_refault_returns"),
                    ("refault_pause_fraction", "prefetch_refault_paused", "prefetch_refault_plans"),
                    ("hot_skip_visits_per_freed_root", "background_hot_skips", "background_roots_freed")):
                    n, d = delta.get(prefix + numerator), delta.get(prefix + denominator)
                    ratios[prefix + name] = n / d if n is not None and d is not None and d > 0 else None
            write_json(out / "summary.json", {
                "exit_code": exit_code, "snapshot_span_seconds": after["monotonic_s"] - before["monotonic_s"],
                "pcie_sampler_exit_code": dmon.returncode if dmon else None,
                "changes": rows, "observed_changes": observed_rows, "derived": ratios,
                "return_distance_histogram": return_distance_histogram(delta),
                "vab_seen": any(k.startswith("vab/") for k in observed),
                "notes": ["Counters are driver-wide, not exclusive to this process.",
                          "Observed changes use first/last available readings; check their coverage times.",
                          "Samples include model loading, warmup, all repeats, and teardown.",
                          "Byte counters describe driver bookkeeping, not measured PCIe wire bytes.",
                          "Rates use the complete before/after interval, not inference-only time."]})
            write_json(out / "run_status.json", {"state": "finished", "monitor_pid": os.getpid(),
                                                "exit_code": exit_code})
            signal.signal(signal.SIGTERM, old_sigterm)
            signal.signal(signal.SIGINT, old_sigint)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == "__main__":
    sys.exit(main())
