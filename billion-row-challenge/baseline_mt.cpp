/*
 * 1-Billion-Row Challenge — Multi-threaded CPU Baseline
 *
 * Memory-maps the file, partitions it across N worker threads
 * (each with its own local hash map), then merges results.
 * Pinned to NUMA node 0 via numactl for optimal memory locality.
 *
 * Build:  g++ -O3 -pthread -o baseline_mt baseline_mt.cpp
 * Usage:  numactl --cpunodebind=0 --membind=0 ./baseline_mt measurements.txt
 *         (or just ./baseline_mt measurements.txt — works without numactl too)
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Stats {
    int min_val;
    int max_val;
    long long sum;
    long long count;
};

static inline int parse_temp(const char *s, int len) {
    bool negative = false;
    int i = 0;
    if (s[0] == '-') {
        negative = true;
        i = 1;
    }
    int result = 0;
    for (; i < len; i++) {
        if (s[i] == '.') continue;
        result = result * 10 + (s[i] - '0');
    }
    return negative ? -result : result;
}

// Each worker processes the byte range [start, end) of the mmap'd file.
// It aligns to line boundaries and aggregates into a local hash map.
static void worker(const char *data, long long start, long long end,
                   long long file_size,
                   std::unordered_map<std::string, Stats> &local_map) {
    local_map.reserve(512);

    // Align start forward to the beginning of a complete line
    // (thread 0 starts at byte 0 which is always a line start).
    if (start > 0) {
        while (start < file_size && data[start - 1] != '\n')
            start++;
    }

    // Align end forward to include the full line that straddles the boundary
    long long actual_end = end;
    if (actual_end < file_size) {
        while (actual_end < file_size && data[actual_end - 1] != '\n')
            actual_end++;
    }

    long long pos = start;
    while (pos < actual_end) {
        // Find end of line
        long long line_start = pos;
        while (pos < file_size && data[pos] != '\n')
            pos++;
        if (pos > file_size) break;

        int line_len = (int)(pos - line_start);
        pos++; // skip '\n'

        if (line_len == 0) continue;

        // Find semicolon
        const char *line = data + line_start;
        int semi = 0;
        while (semi < line_len && line[semi] != ';')
            semi++;
        if (semi >= line_len) continue;

        std::string name(line, semi);
        int temp = parse_temp(line + semi + 1, line_len - semi - 1);

        auto it = local_map.find(name);
        if (it == local_map.end()) {
            local_map[name] = {temp, temp, temp, 1};
        } else {
            auto &st = it->second;
            if (temp < st.min_val) st.min_val = temp;
            if (temp > st.max_val) st.max_val = temp;
            st.sum += temp;
            st.count++;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <measurements.txt> [num_threads]\n", argv[0]);
        return 1;
    }

    auto t_total_start = std::chrono::high_resolution_clock::now();
    auto t_setup_start = t_total_start;
    double setup_s = 0.0;
    double mmap_populate_s = 0.0;
    double process_s = 0.0;
    double finalize_s = 0.0;

    // Default to 22 threads (one Xeon E5-2696 v4 socket = 22 physical cores)
    int num_threads = 22;
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
        if (num_threads < 1) num_threads = 1;
    }

    // Memory-map the file
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    struct stat st;
    fstat(fd, &st);
    long long file_size = st.st_size;

    auto t_setup_end = std::chrono::high_resolution_clock::now();
    setup_s = std::chrono::duration<double>(t_setup_end - t_setup_start).count();

    auto t_mmap_start = std::chrono::high_resolution_clock::now();
    const char *data = (const char *)mmap(nullptr, file_size, PROT_READ,
                                           MAP_PRIVATE | MAP_POPULATE, fd, 0);
    auto t_mmap_end = std::chrono::high_resolution_clock::now();
    mmap_populate_s = std::chrono::duration<double>(t_mmap_end - t_mmap_start).count();
    if (data == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed\n");
        close(fd);
        return 1;
    }
    // Advise sequential access for kernel readahead
    madvise((void *)data, file_size, MADV_SEQUENTIAL);
    close(fd);

    fprintf(stderr, "Multi-threaded CPU: %d threads, file %.1f MB\n",
            num_threads, file_size / (1024.0 * 1024.0));

        auto t_process_start = std::chrono::high_resolution_clock::now();

    // Partition file and launch workers
    std::vector<std::thread> threads(num_threads);
    std::vector<std::unordered_map<std::string, Stats>> local_maps(num_threads);

    long long chunk = file_size / num_threads;
    for (int t = 0; t < num_threads; t++) {
        long long lo = t * chunk;
        long long hi = (t == num_threads - 1) ? file_size : (t + 1) * chunk;
        threads[t] = std::thread(worker, data, lo, hi, file_size,
                                  std::ref(local_maps[t]));
    }

    for (auto &thr : threads)
        thr.join();

    // Merge local maps into global result
    std::unordered_map<std::string, Stats> global;
    global.reserve(512);
    for (auto &lm : local_maps) {
        for (auto &[name, lst] : lm) {
            auto it = global.find(name);
            if (it == global.end()) {
                global[name] = lst;
            } else {
                auto &g = it->second;
                if (lst.min_val < g.min_val) g.min_val = lst.min_val;
                if (lst.max_val > g.max_val) g.max_val = lst.max_val;
                g.sum += lst.sum;
                g.count += lst.count;
            }
        }
    }

    auto t_process_end = std::chrono::high_resolution_clock::now();
    process_s = std::chrono::duration<double>(t_process_end - t_process_start).count();

    auto t_finalize_start = std::chrono::high_resolution_clock::now();

    munmap((void *)data, file_size);

    // Sort and print
    std::vector<std::pair<std::string, Stats>> entries(global.begin(), global.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    printf("{");
    for (size_t i = 0; i < entries.size(); i++) {
        const auto &[name, s] = entries[i];
        double mn = s.min_val / 10.0;
        double mx = s.max_val / 10.0;
        double avg = (s.sum / (double)s.count) / 10.0;
        if (i > 0) printf(", ");
        printf("%s=%.1f/%.1f/%.1f", name.c_str(), mn, avg, mx);
    }
    printf("}\n");

        auto t_finalize_end = std::chrono::high_resolution_clock::now();
        finalize_s = std::chrono::duration<double>(t_finalize_end - t_finalize_start).count();

        auto t_total_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_total_end - t_total_start).count();

    fprintf(stderr, "CPU multi-threaded elapsed: %.3f s  (%d threads)\n",
            elapsed, num_threads);
        fprintf(stderr,
            "CPU multi-threaded phases: setup=%.3f s, mmap_populate=%.3f s, process=%.3f s, finalize=%.3f s\n",
            setup_s, mmap_populate_s, process_s, finalize_s);
    return 0;
}
