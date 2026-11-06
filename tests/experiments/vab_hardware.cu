// SPDX-License-Identifier: MIT
// Linux/CUDA hardware experiment. Does not change driver parameters.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

using Word = unsigned long long;
using Clock = std::chrono::steady_clock;
static constexpr size_t MiB = 1024 * 1024;

static void check(cudaError_t result)
{
    if (result != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(result));
}

__global__ void fill(Word *p, size_t count, Word seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x)
        p[i] = seed ^ Word(i);
}

__global__ void read_hot(const volatile Word *p, size_t count, Word *sink)
{
    Word sum = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x)
        sum += p[i];
    atomicAdd(sink, sum);
}

struct Sample {
    std::map<std::string, Word> numbers;
    std::string raw;
    Word get(const std::string &key) const
    {
        auto it = numbers.find(key);
        if (it == numbers.end())
            throw std::runtime_error("Missing VAB metric: " + key);
        return it->second;
    }
};

static Sample sample(const std::string &path, const char *phase, std::ofstream &log)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot read " + path + "; select the active GPU's VAB proc file");
    Sample s;
    s.raw.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (!in.eof() && in.fail())
        throw std::runtime_error("VAB proc read failed");
    std::istringstream lines(s.raw);
    for (std::string line; std::getline(lines, line);) {
        std::istringstream fields(line);
        std::string key, value, extra;
        if ((fields >> key >> value) && !(fields >> extra)) {
            try {
                size_t used = 0;
                Word number = std::stoull(value, &used, 0);
                if (used == value.size()) s.numbers[key] = number;
            }
            catch (const std::exception &) {}
        }
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
    log << "phase " << phase << " monotonic_ms " << ms << '\n' << s.raw << '\n';
    log.flush();
    if (!log) throw std::runtime_error("Cannot write VAB log");
    return s;
}

static size_t bytes(const char *arg)
{
    size_t used = 0;
    std::string text(arg);
    auto n = std::stoull(text, &used);
    if (text.empty() || text[0] == '-' || used != text.size() || !n || n > SIZE_MAX / MiB)
        throw std::runtime_error("Sizes must be positive integer MiB");
    return size_t(n) * MiB;
}

static void initialize(Word *p, size_t count, Word seed)
{
    fill<<<256, 256>>>(p, count, seed);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
}

static void hold(const std::string &path, const char *phase, std::ofstream &log,
                 Word *hot, size_t count, Word *sink, size_t duration_ms)
{
    auto end = Clock::now() + std::chrono::milliseconds(duration_ms);
    do {
        if (hot) {
            read_hot<<<256, 256>>>(hot, count, sink);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
        }
        sample(path, phase, log);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (Clock::now() < end);
}

static void verify(const Word *p, size_t count, Word seed, const char *name)
{
    // These CPU reads can migrate pages: call only AFTER final counter capture.
    for (size_t i = 0; i < count; ++i)
        if (p[i] != (seed ^ Word(i)))
            throw std::runtime_error(std::string(name) + " data mismatch at word " + std::to_string(i));
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        std::fprintf(stderr, "Usage: %s VAB_PROC HOT_MIB COLD_MIB PRESSURE_MIB LOG_FILE\n"
                             "Uses CUDA device 0 (select with CUDA_VISIBLE_DEVICES).\n", argv[0]);
        return 1;
    }
    try {
        size_t hot_bytes = bytes(argv[2]), cold_bytes = bytes(argv[3]), pressure_bytes = bytes(argv[4]);
        check(cudaSetDevice(0));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0));
        if (!prop.managedMemory || !prop.concurrentManagedAccess)
            throw std::runtime_error("Requires managed memory with concurrentManagedAccess");
        std::printf("GPU %s; sizes MiB: hot=%zu cold=%zu pressure=%zu; L2=%d KiB\n",
                    prop.name, hot_bytes / MiB, cold_bytes / MiB, pressure_bytes / MiB, prop.l2CacheSize / 1024);
        Word *hot, *cold, *pressure, *sink;
        check(cudaMallocManaged(&hot, hot_bytes));
        check(cudaMallocManaged(&cold, cold_bytes));
        check(cudaMallocManaged(&pressure, pressure_bytes));
        check(cudaMalloc(&sink, sizeof(*sink)));
        check(cudaMemset(sink, 0, sizeof(*sink)));
        std::ofstream log(argv[5]);
        if (!log) throw std::runtime_error("Cannot create log file");
        // First managed touch registers this process's GPU with UVM/procfs.
        initialize(hot, 1, 0x1234);
        Sample before = sample(argv[1], "start", log);
        if (!before.get("allocated") || !before.get("supported") || !before.get("sampling_active"))
            throw std::runtime_error("VAB allocation/support/sampling is inactive; see log");
        const size_t interval = std::max<Word>(5, before.get("interval_ms"));
        const size_t phase_ms = std::max<size_t>(2000, 10 * interval);
        hold(argv[1], "idle_before", log, nullptr, 0, sink, phase_ms);
        initialize(cold, cold_bytes / sizeof(Word), 0xc01d);
        initialize(hot, hot_bytes / sizeof(Word), 0x1234);
        hold(argv[1], "hot_before_pressure", log, hot, hot_bytes / sizeof(Word), sink, phase_ms);
        Sample pre_pressure = sample(argv[1], "pre_pressure", log);
        // Materialize pressure gradually; keep hot read between chunks.
        // No cudaFree or CPU data reads occur in the measured pressure interval.
        const size_t batch_words = 64 * MiB / sizeof(Word);
        const size_t pressure_words = pressure_bytes / sizeof(Word);
        for (size_t offset = 0; offset < pressure_words; offset += batch_words) {
            const size_t count = std::min(batch_words, pressure_words - offset);
            // Seed adjusts so every pressure word has the global index pattern.
            // Use a separate seed per chunk, checked the same way below.
            initialize(pressure + offset, count, 0xbeef ^ Word(offset));
            hold(argv[1], "pressure", log, hot, hot_bytes / sizeof(Word), sink, 50);
        }
        hold(argv[1], "hot_after_pressure", log, hot, hot_bytes / sizeof(Word), sink, phase_ms);
        hold(argv[1], "idle_after", log, nullptr, 0, sink, phase_ms);
        Sample after = sample(argv[1], "measurement_end_before_cpu_reads_or_free", log);
        bool healthy = after.get("samples") > before.get("samples") && after.get("sample_valid") &&
                       after.get("enabled_bits") && after.get("dump_errors") == before.get("dump_errors");
        std::printf("Sampling health: %s\n", healthy ? "PASS" : "FAIL (inspect log)");
        for (const char *key : {"samples", "rising_bits", "falling_bits", "selection_changes",
                                "selection_fallbacks", "selected_hot", "selected_cold", "selected_unknown",
                                "background_roots_freed", "background_failures"}) {
            Word a = pre_pressure.get(key), b = after.get(key);
            if (b < a) throw std::runtime_error("Counter reset during test");
            std::printf("pressure delta %-24s %llu\n", key, b - a);
        }
        const bool selection = pre_pressure.get("selection_active") && after.get("selection_active");
        const bool changed = after.get("selection_changes") > pre_pressure.get("selection_changes");
        const bool freed = after.get("background_roots_freed") > pre_pressure.get("background_roots_freed");
        std::printf("VAB changed victim choice: %s\n", selection && changed ? "OBSERVED" : "INCONCLUSIVE");
        std::printf("Completed background root frees: %s\n", freed ? "OBSERVED" : "INCONCLUSIVE");
        std::puts("Aggregate counters do not identify which allocation was evicted; see README.");
        verify(hot, hot_bytes / sizeof(Word), 0x1234, "hot");
        verify(cold, cold_bytes / sizeof(Word), 0xc01d, "cold");
        for (size_t offset = 0; offset < pressure_words; offset += batch_words)
            verify(pressure + offset, std::min(batch_words, pressure_words - offset),
                   0xbeef ^ Word(offset), "pressure");
        std::puts("Full data integrity after pressure: PASS");
        check(cudaFree(hot)); check(cudaFree(cold)); check(cudaFree(pressure)); check(cudaFree(sink));
        return !healthy ? 1 : (selection && changed && freed ? 0 : 2);
    }
    catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
