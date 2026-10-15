# gpubench: measures peak FP32 (TFLOPs/s, CUDA cores) and INT8 (TOPs/s, tensor cores) of the first NVIDIA GPU
# with cuBLAS GEMMs, and prints ONE line: "<host> <gpu-name> <fp32> <int8>". OpenCL-Benchmark is still built in
# but only used when run by hand (docker run --rm --gpus all --entrypoint OpenCL-Benchmark gpubench).
# Build on the target machine (arm64 or amd64):  docker build -t gpubench https://github.com/mahe7998/gpubench.git
# Run:  docker run --rm --gpus all gpubench      (CDI hosts: docker run --rm --device nvidia.com/gpu=all gpubench)
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ make git ca-certificates opencl-headers ocl-icd-opencl-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ProjectPhysX/OpenCL-Benchmark.git /src
WORKDIR /src
RUN mkdir -p bin && g++ src/*.cpp -o bin/OpenCL-Benchmark -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -L./src/OpenCL/lib -lOpenCL
# cuBLAS GEMMs: the measurement used on every host.
COPY cudabench.cu /tmp/cudabench.cu
RUN nvcc -O2 -o /src/bin/cudabench /tmp/cudabench.cu -lcublas -lcublasLt \
    -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends ocl-icd-libopencl1 \
 && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/bin/OpenCL-Benchmark /usr/local/bin/OpenCL-Benchmark
COPY --from=build /src/bin/cudabench /usr/local/bin/cudabench
COPY gpubench.sh /usr/local/bin/gpubench
RUN chmod +x /usr/local/bin/gpubench
ENTRYPOINT ["/usr/local/bin/gpubench"]
