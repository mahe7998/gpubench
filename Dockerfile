# gpubench: measures peak FP32 (TFLOPs/s) and INT8 (TIOPs/s) of the first NVIDIA GPU
# using ProjectPhysX/OpenCL-Benchmark, and prints ONE line: "<host> <gpu-name> <fp32> <int8>".
# Build on the target machine (arm64 or amd64):  docker build -t gpubench https://github.com/mahe7998/gpubench.git
# Run:  docker run --rm --gpus all gpubench      (CDI hosts: docker run --rm --device nvidia.com/gpu=all gpubench)
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ make git ca-certificates opencl-headers ocl-icd-opencl-dev \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ProjectPhysX/OpenCL-Benchmark.git /src
WORKDIR /src
RUN mkdir -p bin && g++ src/*.cpp -o bin/OpenCL-Benchmark -std=c++17 -pthread -O -Wno-comment -I./src/OpenCL/include -L./src/OpenCL/lib -lOpenCL

FROM nvidia/cuda:12.6.3-base-ubuntu24.04
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends ocl-icd-libopencl1 \
 && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/bin/OpenCL-Benchmark /usr/local/bin/OpenCL-Benchmark
COPY gpubench.sh /usr/local/bin/gpubench
RUN chmod +x /usr/local/bin/gpubench
ENTRYPOINT ["/usr/local/bin/gpubench"]
