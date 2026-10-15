#!/bin/sh
# Prints ONE line: <host> <gpu-name> <fp32> <int8> <fp64> <tf32> <fp16> <bf16> <fp8> <fp4> <int4> <formats>  (T(FL)OPS, 0 = unsupported)
# Both numbers come from cuBLAS GEMMs (cudabench) on every machine, so the rows are comparable:
# FP32 = cublasSgemm on the CUDA cores (pedantic math, no TF32), INT8 = cublasGemmEx INT8->INT32 on the tensor cores.
# The GPU name has its spaces replaced by '-' so the line always splits into exactly 4 fields.
HOST="${GPUBENCH_HOST:-$(hostname)}"
CUDA=$(cudabench 2>/tmp/cudabench.err)
if [ -n "$CUDA" ]; then
  echo "$HOST $CUDA"
  exit 0
fi
echo "gpubench: cudabench failed" >&2
cat /tmp/cudabench.err >&2
echo "$HOST none 0 0"
exit 1
