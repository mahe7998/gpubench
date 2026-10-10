#!/bin/sh
# Runs OpenCL-Benchmark on device 0 and prints: <host> <gpu-name> <fp32 TFLOPs/s> <int8 TIOPs/s>
# The GPU name has its spaces replaced by '-' so the line always splits into exactly 4 fields.
HOST="${GPUBENCH_HOST:-$(hostname)}"
# The benchmark overwrites its "Benchmarking ..." lines with carriage returns: turn them into newlines.
OUT=$(echo | OpenCL-Benchmark 0 2>&1 | tr '\r' '\n')
GPU=$(printf '%s\n' "$OUT" | grep 'Device Name' | head -1 | sed -E 's/^\| *Device Name *\| *//; s/ *\|? *$//; s/ /-/g')
FP32=$(printf '%s\n' "$OUT" | grep -E 'FP32 +Compute' | head -1 | sed -E 's/.*\) *([0-9.]+) *TFLOPs\/s.*/\1/')
INT8=$(printf '%s\n' "$OUT" | grep -E 'INT8 +Compute' | head -1 | sed -E 's/.*\) *([0-9.]+) *TIOPs\/s.*/\1/')
if [ -z "$GPU" ] || [ -z "$FP32" ]; then
  echo "gpubench: no OpenCL GPU found" >&2
  printf '%s\n' "$OUT" | tail -20 >&2
  echo "$HOST none 0 0"
  exit 1
fi
echo "$HOST $GPU $FP32 ${INT8:-0}"
