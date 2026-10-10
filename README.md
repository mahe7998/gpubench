# gpubench
Tiny docker image that measures a machine's NVIDIA GPU peak compute with
[OpenCL-Benchmark](https://github.com/ProjectPhysX/OpenCL-Benchmark) and prints one line:

    <host> <gpu-name> <fp32 TFLOPs/s> <int8 TIOPs/s>

Build on the machine itself (works on arm64 and amd64):

    docker build -t gpubench https://github.com/mahe7998/gpubench.git
    docker run --rm --gpus all gpubench            # nvidia runtime
    docker run --rm --device nvidia.com/gpu=all gpubench   # CDI (DGX Spark)

Used by the NLPS setup "test GPU compute on all tailscale active servers with NVIDIA GPU".
