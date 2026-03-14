#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# 1-Billion-Row Challenge — Benchmark Script
#
# Builds both targets, runs them against measurements.txt,
# and prints a side-by-side timing comparison.
#
# Usage:
#   ./benchmark.sh                      # uses measurements.txt
#   ./benchmark.sh path/to/data.txt     # custom data file
# ──────────────────────────────────────────────────────────────
set -euo pipefail

DATA="${1:-measurements.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check data file ──────────────────────────────────────────
if [[ ! -f "$DATA" ]]; then
    echo "Error: data file '$DATA' not found."
    echo ""
    echo "Generate it with:"
    echo "  python3 main.py                           # 1 billion rows"
    echo "  python3 main.py --rows 10000000 -o $DATA  # smaller test"
    exit 1
fi

FILE_SIZE=$(stat --printf="%s" "$DATA" 2>/dev/null || stat -f "%z" "$DATA")
FILE_MB=$(awk "BEGIN { printf \"%.1f\", $FILE_SIZE / 1048576 }")
ROW_COUNT=$(wc -l < "$DATA")

echo "═══════════════════════════════════════════════════════"
echo "  1-Billion-Row Challenge — Benchmark"
echo "═══════════════════════════════════════════════════════"
echo "  Data file : $DATA"
echo "  File size : ${FILE_MB} MB"
echo "  Rows      : ${ROW_COUNT}"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Build ────────────────────────────────────────────────────
echo "Building targets..."
make -s all
echo "Build complete."
echo ""

# ── Run CPU single-threaded baseline ─────────────────────────
echo "─── CPU Baseline (single-threaded) ──────────────────"
./baseline "$DATA" > /tmp/brc_cpu_out.txt 2>/tmp/brc_cpu_time.txt
CPU_TIME=$(grep -oP '[\d.]+(?= s)' /tmp/brc_cpu_time.txt | head -1)
cat /tmp/brc_cpu_time.txt | while read -r line; do echo "  $line"; done
echo ""

# ── Run CPU multi-threaded baseline ──────────────────────────
# Pinned to NUMA node 0 (one Xeon E5-2696 v4 socket, 22 cores)
NUMA_PREFIX=""
if command -v numactl &>/dev/null; then
    NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"
    echo "─── CPU Baseline (multi-threaded, NUMA node 0) ──────"
else
    echo "─── CPU Baseline (multi-threaded) ────────────────────"
fi
$NUMA_PREFIX ./baseline_mt "$DATA" > /tmp/brc_mt_out.txt 2>/tmp/brc_mt_time.txt
MT_TIME=$(grep -oP '[\d.]+(?= s)' /tmp/brc_mt_time.txt | head -1)
cat /tmp/brc_mt_time.txt | while read -r line; do echo "  $line"; done
echo ""

# ── Run CUDA GPU solution ────────────────────────────────────
echo "─── CUDA GPU Solution ─────────────────────────────────"
./solution "$DATA" > /tmp/brc_gpu_out.txt 2>/tmp/brc_gpu_time.txt
GPU_TIME=$(grep -oP '[\d.]+(?= s)' /tmp/brc_gpu_time.txt | head -1)
cat /tmp/brc_gpu_time.txt | while read -r line; do echo "  $line"; done
echo ""

# ── Verify outputs match ────────────────────────────────────
echo "─── Verification ──────────────────────────────────────"
ALL_MATCH=true
if diff -q /tmp/brc_cpu_out.txt /tmp/brc_mt_out.txt > /dev/null 2>&1; then
    echo "  ✓ Single-threaded and multi-threaded CPU outputs match"
else
    echo "  ✗ WARNING: Single-threaded and multi-threaded CPU outputs differ!"
    ALL_MATCH=false
fi
if diff -q /tmp/brc_cpu_out.txt /tmp/brc_gpu_out.txt > /dev/null 2>&1; then
    echo "  ✓ CPU and GPU outputs match"
else
    echo "  ✗ WARNING: CPU and GPU outputs differ!"
    echo "    Run 'diff /tmp/brc_cpu_out.txt /tmp/brc_gpu_out.txt' to inspect."
    ALL_MATCH=false
fi
echo ""

# ── Summary ──────────────────────────────────────────────────
SPEEDUP_MT=$(awk "BEGIN { if ($MT_TIME > 0) printf \"%.2f\", $CPU_TIME / $MT_TIME; else print \"N/A\" }")
SPEEDUP_GPU_VS_ST=$(awk "BEGIN { if ($GPU_TIME > 0) printf \"%.2f\", $CPU_TIME / $GPU_TIME; else print \"N/A\" }")
SPEEDUP_GPU_VS_MT=$(awk "BEGIN { if ($GPU_TIME > 0) printf \"%.2f\", $MT_TIME / $GPU_TIME; else print \"N/A\" }")

echo "═══════════════════════════════════════════════════════"
echo "  Results"
echo "═══════════════════════════════════════════════════════"
printf "  %-28s %10s\n" "" "Time (s)"
printf "  %-28s %10s\n" "────────────────────────────" "──────────"
printf "  %-28s %10s\n" "CPU single-threaded" "$CPU_TIME"
printf "  %-28s %10s\n" "CPU multi-threaded (22 cores)" "$MT_TIME"
printf "  %-28s %10s\n" "CUDA GPU (Tesla V100)" "$GPU_TIME"
printf "  %-28s %10s\n" "────────────────────────────" "──────────"
printf "  %-28s %10sx\n" "MT vs single-thread" "$SPEEDUP_MT"
printf "  %-28s %10sx\n" "GPU vs single-thread" "$SPEEDUP_GPU_VS_ST"
printf "  %-28s %10sx\n" "GPU vs multi-thread" "$SPEEDUP_GPU_VS_MT"
echo "═══════════════════════════════════════════════════════"

# ── Cleanup temp files ───────────────────────────────────────
rm -f /tmp/brc_cpu_out.txt /tmp/brc_mt_out.txt /tmp/brc_gpu_out.txt \
     /tmp/brc_cpu_time.txt /tmp/brc_mt_time.txt /tmp/brc_gpu_time.txt
