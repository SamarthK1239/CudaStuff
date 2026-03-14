/*
 * 1-Billion-Row Challenge — CUDA GPU Solution
 *
 * Reads measurements.txt in chunks via pinned host memory,
 * processes each chunk on the GPU with a massively parallel kernel,
 * and aggregates min/mean/max per station using an open-addressing
 * hash table in device global memory with atomic operations.
 *
 * Target: Tesla V100 (sm_70), 16 GB HBM2
 *
 * Build:  nvcc -O3 -arch=sm_70 -o solution solution.cu
 * Usage:  ./solution measurements.txt
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Hash-table parameters
// ---------------------------------------------------------------------------
// 4096 buckets — load factor ≈ 0.1 for ≤ 413 stations
static constexpr int TABLE_SIZE = 4096;
static constexpr int MAX_NAME_LEN = 100;

// ---------------------------------------------------------------------------
// Per-bucket entry stored in device global memory
// ---------------------------------------------------------------------------
struct __align__(8) HashEntry {
    char  name[MAX_NAME_LEN]; // station name, null-padded
    int   name_len;           // length of the name (excl. null)
    int   min_val;            // temperature × 10  (atomicMin)
    int   max_val;            // temperature × 10  (atomicMax)
    long long sum_val;        // temperature × 10 summed (atomicAdd)
    int   count;              // number of readings   (atomicAdd)
    int   occupied;           // 0 = empty, 1 = taken (atomicCAS)
};

// ---------------------------------------------------------------------------
// FNV-1a hash (device)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int fnv1a(const char *s, int len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Parse temperature from the byte stream.
// Format: optional '-', 1-2 digits, '.', 1 digit
// Returns value × 10 as int.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int parse_temp_gpu(const char *s, int len) {
    bool neg = false;
    int i = 0;
    if (s[0] == '-') {
        neg = true;
        i = 1;
    }
    int val = 0;
    for (; i < len; i++) {
        if (s[i] == '.') continue;
        val = val * 10 + (s[i] - '0');
    }
    return neg ? -val : val;
}

// ---------------------------------------------------------------------------
// Insert / update a station in the hash table (device)
//
// Three-state protocol for the 'occupied' field:
//   0 = empty            — slot is free to claim
//   2 = initializing     — a thread is writing the name + initial stats
//   1 = ready            — name & stats are valid; safe to read/update
//
// Memory ordering is enforced with __threadfence() to guarantee that
// all name bytes and stats written by the claiming thread are globally
// visible before the slot is marked ready.
// ---------------------------------------------------------------------------
__device__ void hash_table_upsert(HashEntry *table, const char *name,
                                   int name_len, int temp_val) {
    unsigned int h = fnv1a(name, name_len);
    unsigned int idx = h & (TABLE_SIZE - 1); // TABLE_SIZE is power of 2

    for (int probe = 0; probe < TABLE_SIZE; probe++) {
        HashEntry *entry = &table[idx];

        // Volatile read of the slot state so the compiler doesn't cache it
        int occ = atomicAdd(&entry->occupied, 0);

        if (occ == 0) {
            // Try to claim: 0 → 2 (initializing)
            int old = atomicCAS(&entry->occupied, 0, 2);
            if (old == 0) {
                // We claimed it — write name bytes
                for (int i = 0; i < name_len; i++)
                    entry->name[i] = name[i];
                entry->name[name_len] = '\0';
                entry->name_len = name_len;

                // Initialize stats
                entry->min_val = temp_val;
                entry->max_val = temp_val;
                entry->sum_val = (long long)temp_val;
                entry->count   = 1;

                // Ensure all writes above are globally visible
                __threadfence();

                // Mark slot as ready (2 → 1)
                atomicExch(&entry->occupied, 1);
                return;
            }
            // Another thread grabbed it — re-read state
            occ = old;
        }

        // Spin until the slot is fully initialized (state == 1)
        while (occ != 1) {
            occ = atomicAdd(&entry->occupied, 0);
        }
        // Ensure we see all data written before the ready flag
        __threadfence();

        // Compare names
        if (entry->name_len == name_len) {
            bool match = true;
            for (int i = 0; i < name_len; i++) {
                if (entry->name[i] != name[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                atomicMin(&entry->min_val, temp_val);
                atomicMax(&entry->max_val, temp_val);
                atomicAdd((unsigned long long *)&entry->sum_val,
                          (unsigned long long)(long long)temp_val);
                atomicAdd(&entry->count, 1);
                return;
            }
        }

        // Linear probe — different station hashed to same bucket
        idx = (idx + 1) & (TABLE_SIZE - 1);
    }
    // Table full — should never happen with 4096 buckets and ≤ 413 stations
}

// ---------------------------------------------------------------------------
// Main per-chunk kernel
//
// Each thread is assigned a byte range [my_start, my_end) within the chunk.
// It aligns forward to the first full line, then processes every complete
// line within its range.
// ---------------------------------------------------------------------------
__global__ void process_chunk_kernel(const char *data, long long chunk_size,
                                      HashEntry *table) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total_threads = (long long)gridDim.x * blockDim.x;

    // Divide chunk evenly among threads
    long long bytes_per_thread = (chunk_size + total_threads - 1) / total_threads;
    long long my_start = tid * bytes_per_thread;
    long long my_end = my_start + bytes_per_thread;
    if (my_start >= chunk_size) return;
    if (my_end > chunk_size) my_end = chunk_size;

    // Align my_start to the beginning of a line:
    // Thread 0 starts at byte 0 (guaranteed line start).
    // All other threads scan forward past the current partial line.
    if (tid != 0) {
        while (my_start < chunk_size && data[my_start - 1] != '\n')
            my_start++;
    }

    // Process all complete lines starting in [my_start, my_end)
    long long pos = my_start;
    while (pos < my_end) {
        // Find end of this line
        long long line_start = pos;
        while (pos < chunk_size && data[pos] != '\n')
            pos++;
        if (pos >= chunk_size) break; // incomplete line at very end — skip

        int line_len = (int)(pos - line_start);
        pos++; // skip '\n'

        // Find semicolon
        int semi = 0;
        while (semi < line_len && data[line_start + semi] != ';')
            semi++;
        if (semi >= line_len) continue; // malformed line

        const char *name = data + line_start;
        int name_len = semi;
        const char *temp_str = data + line_start + semi + 1;
        int temp_len = line_len - semi - 1;

        int temp_val = parse_temp_gpu(temp_str, temp_len);
        hash_table_upsert(table, name, name_len, temp_val);
    }
}

// ---------------------------------------------------------------------------
// Host-side initialization kernel — set min_val to INT_MAX, max_val to INT_MIN
// ---------------------------------------------------------------------------
__global__ void init_table_kernel(HashEntry *table) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TABLE_SIZE) return;

    memset(table[idx].name, 0, MAX_NAME_LEN);
    table[idx].name_len = 0;
    table[idx].min_val  = 2147483647;  // INT_MAX
    table[idx].max_val  = -2147483648; // INT_MIN
    table[idx].sum_val  = 0;
    table[idx].count    = 0;
    table[idx].occupied = 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <measurements.txt>\n", argv[0]);
        return 1;
    }

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    // Get file size
    fseek(fp, 0, SEEK_END);
    long long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    auto wall_start = std::chrono::high_resolution_clock::now();
    auto t_setup_start = wall_start;
    double setup_s = 0.0;
    double read_s = 0.0;
    double finalize_s = 0.0;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;

    cudaEvent_t ev_op_start, ev_op_stop;
    CUDA_CHECK(cudaEventCreate(&ev_op_start));
    CUDA_CHECK(cudaEventCreate(&ev_op_stop));

    // ---- Allocate device hash table ----
    HashEntry *d_table;
    CUDA_CHECK(cudaMalloc(&d_table, TABLE_SIZE * sizeof(HashEntry)));

    // Initialize table on device
    {
        int threads = 256;
        int blocks = (TABLE_SIZE + threads - 1) / threads;
        init_table_kernel<<<blocks, threads>>>(d_table);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Pinned host buffer for chunk I/O ----
    constexpr long long CHUNK_SIZE = 512LL * 1024 * 1024; // 512 MB
    char *h_buf;
    CUDA_CHECK(cudaMallocHost(&h_buf, CHUNK_SIZE));

    // Device buffer for the chunk
    char *d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, CHUNK_SIZE));

    // ---- Leftover handling (partial line at chunk boundary) ----
    char leftover[512];
    int leftover_len = 0;

    auto t_setup_end = std::chrono::high_resolution_clock::now();
    setup_s = std::chrono::duration<double>(t_setup_end - t_setup_start).count();

    long long total_read = 0;
    while (total_read < file_size) {
        // Read a chunk into pinned memory
        long long to_read = CHUNK_SIZE;
        if (leftover_len > 0) {
            // Prepend leftover to the beginning of the host buffer
            memcpy(h_buf, leftover, leftover_len);
            to_read = CHUNK_SIZE - leftover_len;
        }

        auto t_read_start = std::chrono::high_resolution_clock::now();
        long long bytes = (long long)fread(h_buf + leftover_len, 1, to_read, fp);
        auto t_read_end = std::chrono::high_resolution_clock::now();
        read_s += std::chrono::duration<double>(t_read_end - t_read_start).count();

        long long chunk_len = leftover_len + bytes;
        leftover_len = 0;
        total_read += bytes;

        if (chunk_len == 0) break;

        // If this is NOT the last chunk, trim to the last complete line
        if (total_read < file_size) {
            long long trim = chunk_len - 1;
            while (trim >= 0 && h_buf[trim] != '\n') trim--;
            if (trim >= 0) {
                // Save the partial tail as leftover
                leftover_len = (int)(chunk_len - trim - 1);
                memcpy(leftover, h_buf + trim + 1, leftover_len);
                chunk_len = trim + 1; // include the '\n'
            }
        }

        // Copy chunk to device
        CUDA_CHECK(cudaEventRecord(ev_op_start));
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, chunk_len, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(ev_op_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_op_stop));
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_op_start, ev_op_stop));
            h2d_ms += ms;
        }

        // Launch kernel
        // Heuristic: 1 thread per ~512 bytes → gives each thread ~1-3 lines
        int threads_per_block = 256;
        long long total_threads = (chunk_len + 511) / 512;
        if (total_threads < 256) total_threads = 256;
        int blocks = (int)((total_threads + threads_per_block - 1) / threads_per_block);
        // Cap grid size at 65535 blocks (safe for all architectures)
        if (blocks > 65535) blocks = 65535;

        CUDA_CHECK(cudaEventRecord(ev_op_start));
        process_chunk_kernel<<<blocks, threads_per_block>>>(d_buf, chunk_len, d_table);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_op_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_op_stop));
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_op_start, ev_op_stop));
            kernel_ms += ms;
        }

        // Progress
        fprintf(stderr, "  Processed %lld / %lld bytes (%.1f%%)\r",
                total_read, file_size, total_read * 100.0 / file_size);
    }
    fprintf(stderr, "\n");

    fclose(fp);
    CUDA_CHECK(cudaFreeHost(h_buf));
    CUDA_CHECK(cudaFree(d_buf));

    // ---- Copy hash table back to host ----
    auto t_finalize_start = std::chrono::high_resolution_clock::now();

    HashEntry *h_table = new HashEntry[TABLE_SIZE];
    CUDA_CHECK(cudaEventRecord(ev_op_start));
    CUDA_CHECK(cudaMemcpy(h_table, d_table, TABLE_SIZE * sizeof(HashEntry),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_op_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_op_stop));
    {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_op_start, ev_op_stop));
        d2h_ms += ms;
    }
    CUDA_CHECK(cudaFree(d_table));

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();

    // ---- Collect occupied entries and sort ----
    struct Result {
        std::string name;
        int min_val, max_val;
        long long sum_val;
        int count;
    };
    std::vector<Result> results;
    results.reserve(512);

    for (int i = 0; i < TABLE_SIZE; i++) {
        if (h_table[i].occupied && h_table[i].count > 0) {
            results.push_back({
                std::string(h_table[i].name, h_table[i].name_len),
                h_table[i].min_val,
                h_table[i].max_val,
                h_table[i].sum_val,
                h_table[i].count,
            });
        }
    }
    delete[] h_table;

    std::sort(results.begin(), results.end(),
              [](const Result &a, const Result &b) { return a.name < b.name; });

    // ---- Print output (same format as CPU baseline) ----
    printf("{");
    for (size_t i = 0; i < results.size(); i++) {
        const auto &r = results[i];
        double mn  = r.min_val / 10.0;
        double mx  = r.max_val / 10.0;
        double avg = (r.sum_val / (double)r.count) / 10.0;
        if (i > 0) printf(", ");
        printf("%s=%.1f/%.1f/%.1f", r.name.c_str(), mn, avg, mx);
    }
    printf("}\n");

        auto t_finalize_end = std::chrono::high_resolution_clock::now();
        finalize_s = std::chrono::duration<double>(t_finalize_end - t_finalize_start).count();

        double h2d_s = h2d_ms / 1000.0;
        double kernel_s = kernel_ms / 1000.0;
        double d2h_s = d2h_ms / 1000.0;
        double device_active_s = h2d_s + kernel_s + d2h_s;

        fprintf(stderr, "CUDA GPU elapsed: %.3f s\n", wall_s);
        fprintf(stderr,
            "CUDA GPU phases: setup=%.3f s, read=%.3f s, h2d=%.3f s, kernel=%.3f s, d2h=%.3f s, finalize=%.3f s\n",
            setup_s, read_s, h2d_s, kernel_s, d2h_s, finalize_s);
        fprintf(stderr, "CUDA GPU device-active: %.3f s\n", device_active_s);

        CUDA_CHECK(cudaEventDestroy(ev_op_start));
        CUDA_CHECK(cudaEventDestroy(ev_op_stop));

    return 0;
}
