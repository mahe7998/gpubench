// cudabench: measures the first NVIDIA GPU with cuBLAS / cuBLASLt GEMMs and prints ONE line:
//   <gpu-name-with-dashes> <fp32> <int8> <fp64> <tf32> <fp16> <bf16> <fp8> <fp4> <int4> <formats>
// Units: T(FL)OPS. fp32 = cublasSgemm with pedantic math (CUDA cores), the others are tensor-core GEMMs.
// A format is "supported" when cublasLtMatmulAlgoGetHeuristic returns an algorithm for it AND the matmul runs;
// unsupported formats print 0. <formats> is the comma-separated list of the supported ones.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define CK(x) do { if ((x) != cudaSuccess) { fprintf(stderr, "cudabench: %s failed: %s\n", #x, cudaGetErrorString(cudaGetLastError())); return 1; } } while (0)
#define CB(x) do { if ((x) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cudabench: %s failed (%d)\n", #x, (int)(x)); return 1; } } while (0)

static const int N = 8192, WARM = 2, ITERS = 10;
static cudaEvent_t g_t0, g_t1;

static double tops(float ms) {
    return 2.0 * (double) N * N * N * ITERS / (ms / 1000.0) / 1e12;
}

struct Probe {
    const char* name;
    cudaDataType ab, cd;          // A/B element type, C/D element type
    cublasComputeType_t compute;
    cudaDataType scale;
    int bits_ab, bits_cd;
    bool block_scale;             // FP4: VEC16_UE4M3 block scaling is mandatory
    cublasOperation_t opa, opb;   // FP8/FP4/INT4 need TN
};

// Runs one cuBLASLt matmul configuration; returns T(FL)OPS, or 0 when the GPU / library does not support it.
static double lt_probe(cublasLtHandle_t lt, const Probe& p, void* ws, size_t wsz) {
    double result = 0;
    size_t sz_ab = (size_t) N * N * p.bits_ab / 8, sz_cd = (size_t) N * N * p.bits_cd / 8;
    void *a = NULL, *b = NULL, *c = NULL, *sa = NULL, *sb = NULL;
    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t la = NULL, lb = NULL, lc = NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heur = {};
    int nheur = 0;
    const float one = 1.0f, zero = 0.0f;
    const int ione = 1, izero = 0;
    const void* alpha = (p.scale == CUDA_R_32I) ? (const void*) &ione : (const void*) &one;
    const void* beta = (p.scale == CUDA_R_32I) ? (const void*) &izero : (const void*) &zero;

    if (cudaMalloc(&a, sz_ab) != cudaSuccess || cudaMalloc(&b, sz_ab) != cudaSuccess || cudaMalloc(&c, sz_cd) != cudaSuccess) goto done;
    cudaMemset(a, 0x11, sz_ab); cudaMemset(b, 0x11, sz_ab);
    if (cublasLtMatmulDescCreate(&op, p.compute, p.scale) != CUBLAS_STATUS_SUCCESS) goto done;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &p.opa, sizeof(p.opa));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &p.opb, sizeof(p.opb));
    if (p.block_scale) {
        // one UE4M3 scale per 16 elements along the inner dimension (layout is tiled; contents do not matter for timing)
        size_t sz_scale = (size_t) N * N / 16 + 4096;
        if (cudaMalloc(&sa, sz_scale) != cudaSuccess || cudaMalloc(&sb, sz_scale) != cudaSuccess) goto done;
        cudaMemset(sa, 0x38, sz_scale); cudaMemset(sb, 0x38, sz_scale);  // 0x38 = 1.0 in E4M3
        cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb));
    }
    if (cublasLtMatrixLayoutCreate(&la, p.ab, N, N, N) != CUBLAS_STATUS_SUCCESS) goto done;
    if (cublasLtMatrixLayoutCreate(&lb, p.ab, N, N, N) != CUBLAS_STATUS_SUCCESS) goto done;
    if (cublasLtMatrixLayoutCreate(&lc, p.cd, N, N, N) != CUBLAS_STATUS_SUCCESS) goto done;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) goto done;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &heur, &nheur) != CUBLAS_STATUS_SUCCESS || nheur == 0) {
        fprintf(stderr, "cudabench: %s: no cuBLASLt algorithm (unsupported)\n", p.name);
        goto done;
    }
    {
        cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
        for (int i = 0; i < WARM && st == CUBLAS_STATUS_SUCCESS; i++)
            st = cublasLtMatmul(lt, op, alpha, a, la, b, lb, beta, c, lc, c, lc, &heur.algo, ws, wsz, 0);
        if (st != CUBLAS_STATUS_SUCCESS || cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "cudabench: %s: matmul failed (%d / %s)\n", p.name, (int) st, cudaGetErrorString(cudaGetLastError()));
            goto done;
        }
        cudaEventRecord(g_t0);
        for (int i = 0; i < ITERS; i++)
            cublasLtMatmul(lt, op, alpha, a, la, b, lb, beta, c, lc, c, lc, &heur.algo, ws, wsz, 0);
        cudaEventRecord(g_t1); cudaEventSynchronize(g_t1);
        float ms = 0; cudaEventElapsedTime(&ms, g_t0, g_t1);
        result = tops(ms);
    }
done:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (op) cublasLtMatmulDescDestroy(op);
    cudaFree(sa); cudaFree(sb); cudaFree(a); cudaFree(b); cudaFree(c);
    cudaGetLastError();  // clear a sticky error from an unsupported attempt
    return result;
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::string name = prop.name;
    for (auto& ch : name) if (ch == ' ') ch = '-';
    CK(cudaEventCreate(&g_t0)); CK(cudaEventCreate(&g_t1));

    cublasHandle_t h;
    CB(cublasCreate(&h));
    float ms = 0;

    // FP32 (CUDA cores, no TF32) and TF32 (tensor cores): C = A * B with cublasSgemm
    float *a, *b, *c;
    CK(cudaMalloc(&a, (size_t) N * N * sizeof(float)));
    CK(cudaMalloc(&b, (size_t) N * N * sizeof(float)));
    CK(cudaMalloc(&c, (size_t) N * N * sizeof(float)));
    CK(cudaMemset(a, 0, (size_t) N * N * sizeof(float)));
    CK(cudaMemset(b, 0, (size_t) N * N * sizeof(float)));
    const float one = 1.0f, zero = 0.0f;
    double fp32 = 0, tf32 = 0;
    for (int pass = 0; pass < 2; pass++) {
        CB(cublasSetMathMode(h, pass == 0 ? CUBLAS_PEDANTIC_MATH : CUBLAS_TF32_TENSOR_OP_MATH));
        for (int i = 0; i < WARM; i++)
            CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, a, N, b, N, &zero, c, N));
        CK(cudaEventRecord(g_t0));
        for (int i = 0; i < ITERS; i++)
            CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, a, N, b, N, &zero, c, N));
        CK(cudaEventRecord(g_t1)); CK(cudaEventSynchronize(g_t1));
        CK(cudaEventElapsedTime(&ms, g_t0, g_t1));
        (pass == 0 ? fp32 : tf32) = tops(ms);
    }
    cudaFree(a); cudaFree(b); cudaFree(c);
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    // FP64: cublasDgemm
    double *da, *db, *dc, fp64 = 0;
    CK(cudaMalloc(&da, (size_t) N * N * sizeof(double)));
    CK(cudaMalloc(&db, (size_t) N * N * sizeof(double)));
    CK(cudaMalloc(&dc, (size_t) N * N * sizeof(double)));
    CK(cudaMemset(da, 0, (size_t) N * N * sizeof(double)));
    CK(cudaMemset(db, 0, (size_t) N * N * sizeof(double)));
    {
        const double done = 1.0, dzero = 0.0;
        for (int i = 0; i < WARM; i++)
            CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &done, da, N, db, N, &dzero, dc, N));
        CK(cudaEventRecord(g_t0));
        for (int i = 0; i < ITERS / 2; i++)   // FP64 is slow on consumer parts: half the iterations
            CB(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &done, da, N, db, N, &dzero, dc, N));
        CK(cudaEventRecord(g_t1)); CK(cudaEventSynchronize(g_t1));
        CK(cudaEventElapsedTime(&ms, g_t0, g_t1));
        fp64 = tops(ms) / 2;
    }
    cudaFree(da); cudaFree(db); cudaFree(dc);

    // INT8 with INT32 accumulation (cublasGemmEx, tensor cores)
    signed char *ia, *ib; int *ic; double int8 = 0;
    CK(cudaMalloc(&ia, (size_t) N * N));
    CK(cudaMalloc(&ib, (size_t) N * N));
    CK(cudaMalloc(&ic, (size_t) N * N * sizeof(int)));
    CK(cudaMemset(ia, 1, (size_t) N * N));
    CK(cudaMemset(ib, 1, (size_t) N * N));
    {
        const int ione = 1, izero = 0;
        cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
        for (int i = 0; i < WARM && st == CUBLAS_STATUS_SUCCESS; i++)
            st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &ione, ia, CUDA_R_8I, N, ib, CUDA_R_8I, N,
                              &izero, ic, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        if (st == CUBLAS_STATUS_SUCCESS) {
            CK(cudaEventRecord(g_t0));
            for (int i = 0; i < ITERS; i++)
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &ione, ia, CUDA_R_8I, N, ib, CUDA_R_8I, N,
                             &izero, ic, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
            CK(cudaEventRecord(g_t1)); CK(cudaEventSynchronize(g_t1));
            CK(cudaEventElapsedTime(&ms, g_t0, g_t1));
            int8 = tops(ms);
        } else
            fprintf(stderr, "cudabench: INT8 GEMM not supported on this GPU (%d)\n", (int) st);
    }
    cudaFree(ia); cudaFree(ib); cudaFree(ic);
    cublasDestroy(h);

    // Tensor-core formats through cuBLASLt: supported when the heuristic finds an algorithm and the matmul runs.
    cublasLtHandle_t lt;
    CB(cublasLtCreate(&lt));
    void* ws = NULL; size_t wsz = 256u << 20;
    CK(cudaMalloc(&ws, wsz));
    const Probe probes[] = {
        {"fp16", CUDA_R_16F,     CUDA_R_16F,  CUBLAS_COMPUTE_32F, CUDA_R_32F, 16, 16, false, CUBLAS_OP_N, CUBLAS_OP_N},
        {"bf16", CUDA_R_16BF,    CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, 16, 16, false, CUBLAS_OP_N, CUBLAS_OP_N},
        {"fp8",  CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,  8, 16, false, CUBLAS_OP_T, CUBLAS_OP_N},
#if CUDART_VERSION >= 12080   // CUDA_R_4F_E2M1 exists from CUDA 12.8
        {"fp4",  CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,  4, 16, true,  CUBLAS_OP_T, CUBLAS_OP_N},
#endif
        {"int4", CUDA_R_4I,      CUDA_R_32I,  CUBLAS_COMPUTE_32I, CUDA_R_32I,  4, 32, false, CUBLAS_OP_T, CUBLAS_OP_N},
    };
    double fp16 = 0, bf16 = 0, fp8 = 0, fp4 = 0, int4 = 0;
    for (const Probe& p : probes) {
        double v = lt_probe(lt, p, ws, wsz);
        std::string nm = p.name;
        if (nm == "fp16") fp16 = v; else if (nm == "bf16") bf16 = v; else if (nm == "fp8") fp8 = v;
        else if (nm == "fp4") fp4 = v; else if (nm == "int4") int4 = v;
    }
    cudaFree(ws);
    cublasLtDestroy(lt);

    std::string formats;
    struct { const char* n; double v; } all[] = {{"fp64", fp64}, {"fp32", fp32}, {"tf32", tf32}, {"fp16", fp16}, {"bf16", bf16},
                                                  {"fp8", fp8}, {"fp4", fp4}, {"int8", int8}, {"int4", int4}};
    for (auto& f : all) if (f.v > 0) { if (!formats.empty()) formats += ","; formats += f.n; }
    printf("%s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %s\n", name.c_str(),
           fp32, int8, fp64, tf32, fp16, bf16, fp8, fp4, int4, formats.empty() ? "none" : formats.c_str());
    return 0;
}
