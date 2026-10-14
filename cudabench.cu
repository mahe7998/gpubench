// cudabench: CUDA fallback of gpubench for hosts without NVIDIA OpenCL (Docker Desktop / WSL2).
// Prints ONE line: "<gpu-name-with-dashes> <fp32 TFLOPs/s> <int8 TOPs/s>" using cuBLAS GEMMs
// (FP32 SGEMM, INT8 GEMM with INT32 accumulation on tensor cores when available).
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do { if ((x) != cudaSuccess) { fprintf(stderr, "cudabench: %s failed: %s\n", #x, cudaGetErrorString(cudaGetLastError())); return 1; } } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cudabench: %s failed (%d)\n", #x, (int)(x)); return 1; } } while (0)

static double tflops(int n, int iters, float ms) {
    return 2.0 * (double) n * n * n * iters / (ms / 1000.0) / 1e12;
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    std::string name = prop.name;
    for (auto& c : name) if (c == ' ') c = '-';

    const int n = 8192, warm = 2, iters = 10;
    cublasHandle_t h;
    CB(cublasCreate(&h));
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    // FP32: C = A * B
    float *a, *b, *c;
    CK(cudaMalloc(&a, (size_t) n * n * sizeof(float)));
    CK(cudaMalloc(&b, (size_t) n * n * sizeof(float)));
    CK(cudaMalloc(&c, (size_t) n * n * sizeof(float)));
    CK(cudaMemset(a, 0, (size_t) n * n * sizeof(float)));
    CK(cudaMemset(b, 0, (size_t) n * n * sizeof(float)));
    const float one = 1.0f, zero = 0.0f;
    CB(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH)); // true FP32, no TF32
    for (int i = 0; i < warm; i++)
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, a, n, b, n, &zero, c, n));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++)
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, a, n, b, n, &zero, c, n));
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
    double fp32 = tflops(n, iters, ms);
    cudaFree(a); cudaFree(b); cudaFree(c);

    // INT8 with INT32 accumulation: C(int32) = A^T(int8) * B(int8)
    signed char *ia, *ib; int *ic;
    CK(cudaMalloc(&ia, (size_t) n * n));
    CK(cudaMalloc(&ib, (size_t) n * n));
    CK(cudaMalloc(&ic, (size_t) n * n * sizeof(int)));
    CK(cudaMemset(ia, 1, (size_t) n * n));
    CK(cudaMemset(ib, 1, (size_t) n * n));
    const int ione = 1, izero = 0;
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
    double int8 = 0;
    cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < warm && st == CUBLAS_STATUS_SUCCESS; i++)
        st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &ione, ia, CUDA_R_8I, n, ib, CUDA_R_8I, n,
                          &izero, ic, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) {
        CK(cudaEventRecord(t0));
        for (int i = 0; i < iters; i++)
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n, &ione, ia, CUDA_R_8I, n, ib, CUDA_R_8I, n,
                         &izero, ic, CUDA_R_32I, n, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        CK(cudaEventElapsedTime(&ms, t0, t1));
        int8 = tflops(n, iters, ms);
    } else
        fprintf(stderr, "cudabench: INT8 GEMM not supported on this GPU (%d)\n", (int) st);
    cudaFree(ia); cudaFree(ib); cudaFree(ic);
    cublasDestroy(h);
    printf("%s %.3f %.3f\n", name.c_str(), fp32, int8);
    return 0;
}
