#!/bin/sh
# Runs OpenCL-Benchmark on device 0 and prints: <host> <gpu-name> <fp32 TFLOPs/s> <int8 TIOPs/s>
# GPU name has its spaces replaced by '-' so the line splits into exactly 4 fields.
HOST="${GPUBENCH_HOST:-$(hostname)}"
OUT=$(echo | OpenCL-Benchmark 0 2>&1)
GPU=$(printf '%s\n' "$OUT" | sed -n 's/.*Device Name[^A-Za-z0-9]*\(.*\)$/\1/p' | head -1 | sed 's/[[:space:]]*$//; s/ /-/g')
FP32=$(printf '%s\n' "$OUT" | grep -E '^\|? *FP32 +Compute' | sed -E 's/.*\) *([0-9.]+) *TFLOPs\/s.*/\1/' | head -1)
INT8=$(printf '%s\n' "$OUT" | grep -E '^\|? *INT8 +Compute' | sed -E 's/.*\) *([0-9.]+) *TIOPs\/s.*/\1/' | head -1)
if [ -z "$GPU" ] || [ -z "$FP32" ]; then
  echo "gpubench: no OpenCL GPU found" >&2
  printf '%s\n' "$OUT" | tail -20 >&2
  echo "$HOST none 0 0"
  exit 1
fi
echo "$HOST ${GPU:-unknown} ${FP32:-0} ${INT8:-0}"
