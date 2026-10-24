"""Exercise the real demand-first queue against the real coalesced prefetch policy."""
from pathlib import Path
import runpy
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    # Reuse only the mock external GPU/kernel APIs. Neither production path is
    # replaced: the queue flush calls the extracted coalesced policy directly.
    mock = runpy.run_path(str(Path(__file__).with_name("test_coalesced_prefetch.py")))
    policy = (ROOT / "kernel-open/nvidia-uvm/uvm_cpu_block_policy.c").read_text()
    first = policy.index("typedef struct\n{\n    size_t first;")
    last = policy.index("void uvm_cpu_block_policy_rearm_on_eviction", first)
    notifications = (ROOT / "kernel-open/nvidia-uvm/uvm_gpu_access_counters.c").read_text()
    queue_start = notifications.index("#define UVM_DEMAND_FIRST_PREFETCH_QUEUE_CAPACITY")
    queue_end = notifications.index("// Each page in a tracked physical range", queue_start)
    scenarios = r'''
static void reset(void) {
    submitted=waits=ends=admitted=freed=allocations=0;
    range.cpu_prefetch_lock=0;
    for (unsigned i=0;i<ARRAY_SIZE(range.cpu_prefetch_windows);++i)
        range.cpu_prefetch_windows[i].gpu=NULL;
    for (int i=0;i<4;++i)
        blocks[i]=(uvm_va_block_t){.managed_range=&range,.start=i*UVM_VA_BLOCK_SIZE};
}
static void assert_no_owned_window(void) {
    assert(!admitted && !range.cpu_prefetch_lock);
    for (unsigned i=0;i<ARRAY_SIZE(range.cpu_prefetch_windows);++i)
        assert(!range.cpu_prefetch_windows[i].gpu);
}
int main(void) {
    uvm_gpu_t gpu={0}, other_gpu={0};
    demand_first_prefetch_queue_t queue={0};

    // Queueing does not initiate prefetch; duplicate triggers remain cheap.
    reset();mode=0;
    demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    assert(queue.count==2 && !submitted && !admitted);
    demand_first_prefetch_queue_flush(&queue);
    assert(!queue.count && submitted==4 && waits==1 && ends==1);
    assert_no_owned_window();

    // Bounded flushes use the actual policy, including its one-shot tracking.
    reset();mode=0;
    for (unsigned i=0;i<=UVM_DEMAND_FIRST_PREFETCH_QUEUE_CAPACITY;++i)
        demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    assert(queue.count==1 && submitted==4);
    demand_first_prefetch_queue_flush(&queue);
    assert(!queue.count && submitted==4 && waits==1);
    assert_no_owned_window();

    // Another buffer's older overlap remains visible through queued dispatch.
    reset();mode=7;
    range.cpu_prefetch_windows[0].gpu=&gpu;
    range.cpu_prefetch_windows[0].first=1;
    range.cpu_prefetch_windows[0].outer=3;
    demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    assert(!submitted);
    demand_first_prefetch_queue_flush(&queue);
    assert(submitted==2 && waits==1 && blocks[0].gpu_prefetch_retry);
    assert_no_owned_window();

    // Full overlap stays retryable and does not wait for an empty tracker.
    reset();mode=8;
    range.cpu_prefetch_windows[0].gpu=&gpu;
    range.cpu_prefetch_windows[0].first=0;
    range.cpu_prefetch_windows[0].outer=4;
    demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    demand_first_prefetch_queue_flush(&queue);
    assert(!submitted && !waits && ends==1 && blocks[0].gpu_prefetch_retry);
    prefetch_window_end(&range,0);
    assert_no_owned_window();

    // GPU identity still isolates coalescing when dispatch comes from a queue.
    reset();mode=9;
    range.cpu_prefetch_windows[0].gpu=&other_gpu;
    range.cpu_prefetch_windows[0].first=0;
    range.cpu_prefetch_windows[0].outer=4;
    demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    demand_first_prefetch_queue_flush(&queue);
    assert(submitted==4 && waits==1);
    prefetch_window_end(&range,0);
    assert_no_owned_window();

    // Demand-error discard does not create a coverage window or admission.
    reset();mode=0;
    demand_first_prefetch_queue_push(&queue,&blocks[0],&gpu,NULL);
    demand_first_prefetch_queue_discard(&queue);
    assert(!queue.count && !submitted && !waits && !ends);
    assert_no_owned_window();
    puts("Combined queue/coalescing production integration checks passed.");
}
'''
    source = (mock["HARNESS"] + "\ntypedef unsigned NvU32;\n" + policy[first:last]
              + notifications[queue_start:queue_end] + scenarios)
    with tempfile.TemporaryDirectory(prefix="uvm-combined-") as directory:
        path = Path(directory)
        (path / "test.c").write_text(source)
        subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined", str(path / "test.c"),
                        "-o", str(path / "test")], check=True)
        subprocess.run([str(path / "test")], check=True)


if __name__ == "__main__":
    main()
