/*
 * 1-Billion-Row Challenge — Single-threaded CPU Baseline
 *
 * Reads measurements.txt line by line using a large fread buffer,
 * aggregates min/mean/max per station in an unordered_map,
 * and prints the sorted results to stdout.
 * Elapsed wall-clock time is printed to stderr.
 *
 * Build:  g++ -O3 -o baseline baseline.cpp
 * Usage:  ./baseline measurements.txt
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct Stats {
    int min_val;   // temperature × 10
    int max_val;   // temperature × 10
    long long sum; // temperature × 10, accumulated
    long long count;
};

// Parse a temperature string like "12.3", "-5.7", "0.0" into an int × 10.
// Expects exactly one decimal digit after the dot.
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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <measurements.txt>\n", argv[0]);
        return 1;
    }

    auto t_total_start = std::chrono::high_resolution_clock::now();
    auto t_setup_start = t_total_start;
    double setup_s = 0.0;
    double read_s = 0.0;
    double parse_s = 0.0;
    double finalize_s = 0.0;

    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", argv[1]);
        return 1;
    }

    // 64 MB read buffer
    constexpr size_t BUF_SIZE = 64 * 1024 * 1024;
    char *buf = new char[BUF_SIZE];

    std::unordered_map<std::string, Stats> map;
    map.reserve(512);

    // Leftover bytes from previous fread that didn't end on a newline
    char leftover[256];
    int leftover_len = 0;

    auto t_setup_end = std::chrono::high_resolution_clock::now();
    setup_s = std::chrono::duration<double>(t_setup_end - t_setup_start).count();

    size_t bytes_read;
    while (true) {
        auto t_read_start = std::chrono::high_resolution_clock::now();
        bytes_read = fread(buf, 1, BUF_SIZE, fp);
        auto t_read_end = std::chrono::high_resolution_clock::now();
        read_s += std::chrono::duration<double>(t_read_end - t_read_start).count();

        if (bytes_read == 0) {
            break;
        }

        auto t_parse_start = std::chrono::high_resolution_clock::now();
        size_t start = 0;

        // If we have leftover from previous chunk, prepend it
        // by finding the first newline and combining.
        if (leftover_len > 0) {
            // Find first newline in buf
            size_t nl = 0;
            while (nl < bytes_read && buf[nl] != '\n') nl++;

            // Build complete line: leftover + buf[0..nl)
            // leftover buffer is 256 bytes and station names are ≤100 chars + temp ≤6 chars
            char line[512];
            memcpy(line, leftover, leftover_len);
            if (nl < bytes_read) {
                memcpy(line + leftover_len, buf, nl);
                int line_len = leftover_len + (int)nl;
                leftover_len = 0;
                start = nl + 1;

                // Parse this line
                int semi = 0;
                while (semi < line_len && line[semi] != ';') semi++;
                if (semi < line_len) {
                    std::string name(line, semi);
                    int temp = parse_temp(line + semi + 1, line_len - semi - 1);
                    auto it = map.find(name);
                    if (it == map.end()) {
                        map[name] = {temp, temp, temp, 1};
                    } else {
                        auto &st = it->second;
                        if (temp < st.min_val) st.min_val = temp;
                        if (temp > st.max_val) st.max_val = temp;
                        st.sum += temp;
                        st.count++;
                    }
                }
            } else {
                // Entire chunk has no newline — append to leftover (shouldn't happen with real data)
                memcpy(leftover + leftover_len, buf, bytes_read);
                leftover_len += (int)bytes_read;
                continue;
            }
        }

        // Process complete lines in buf[start..bytes_read)
        for (size_t i = start; i < bytes_read; i++) {
            if (buf[i] == '\n') {
                int line_len = (int)(i - start);
                const char *line = buf + start;

                // Find semicolon
                int semi = 0;
                while (semi < line_len && line[semi] != ';') semi++;

                if (semi < line_len) {
                    std::string name(line, semi);
                    int temp = parse_temp(line + semi + 1, line_len - semi - 1);
                    auto it = map.find(name);
                    if (it == map.end()) {
                        map[name] = {temp, temp, temp, 1};
                    } else {
                        auto &st = it->second;
                        if (temp < st.min_val) st.min_val = temp;
                        if (temp > st.max_val) st.max_val = temp;
                        st.sum += temp;
                        st.count++;
                    }
                }

                start = i + 1;
            }
        }

        // Save leftover (partial line at end of buffer)
        if (start < bytes_read) {
            leftover_len = (int)(bytes_read - start);
            memcpy(leftover, buf + start, leftover_len);
        } else {
            leftover_len = 0;
        }

        auto t_parse_end = std::chrono::high_resolution_clock::now();
        parse_s += std::chrono::duration<double>(t_parse_end - t_parse_start).count();
    }

    // Handle any final leftover (file didn't end with newline)
    if (leftover_len > 0) {
        auto t_parse_start = std::chrono::high_resolution_clock::now();
        int semi = 0;
        while (semi < leftover_len && leftover[semi] != ';') semi++;
        if (semi < leftover_len) {
            std::string name(leftover, semi);
            int temp = parse_temp(leftover + semi + 1, leftover_len - semi - 1);
            auto it = map.find(name);
            if (it == map.end()) {
                map[name] = {temp, temp, temp, 1};
            } else {
                auto &st = it->second;
                if (temp < st.min_val) st.min_val = temp;
                if (temp > st.max_val) st.max_val = temp;
                st.sum += temp;
                st.count++;
            }
        }
        auto t_parse_end = std::chrono::high_resolution_clock::now();
        parse_s += std::chrono::duration<double>(t_parse_end - t_parse_start).count();
    }

    fclose(fp);
    delete[] buf;

    auto t_finalize_start = std::chrono::high_resolution_clock::now();

    // Sort by station name and print results
    std::vector<std::pair<std::string, Stats>> entries(map.begin(), map.end());
    std::sort(entries.begin(), entries.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    printf("{");
    for (size_t i = 0; i < entries.size(); i++) {
        const auto &[name, st] = entries[i];
        double mn = st.min_val / 10.0;
        double mx = st.max_val / 10.0;
        double avg = (st.sum / (double)st.count) / 10.0;
        if (i > 0) printf(", ");
        printf("%s=%.1f/%.1f/%.1f", name.c_str(), mn, avg, mx);
    }
    printf("}\n");

    auto t_finalize_end = std::chrono::high_resolution_clock::now();
    finalize_s = std::chrono::duration<double>(t_finalize_end - t_finalize_start).count();

    auto t_total_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_total_end - t_total_start).count();

    fprintf(stderr, "CPU baseline elapsed: %.3f s\n", elapsed);
    fprintf(stderr,
            "CPU baseline phases: setup=%.3f s, read=%.3f s, parse=%.3f s, finalize=%.3f s\n",
            setup_s, read_s, parse_s, finalize_s);
    return 0;
}
