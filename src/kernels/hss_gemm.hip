#include "kernels/hss_gemm.h"

#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int TILE = 16;
constexpr int NT_BLOCK_M = 64;
constexpr int NT_BLOCK_N = 64;
constexpr int NT_BLOCK_K = 16;
constexpr int NT_THREAD_TILE_M = 4;
constexpr int NT_THREAD_TILE_N = 4;
constexpr int NT_THREADS_X = NT_BLOCK_M / NT_THREAD_TILE_M;
constexpr int NT_THREADS_Y = NT_BLOCK_N / NT_THREAD_TILE_N;
constexpr int NT_THREADS = NT_THREADS_X * NT_THREADS_Y;

constexpr int TUNED_BLOCK_M = 128;
constexpr int TUNED_BLOCK_N = 64;
constexpr int TUNED_BLOCK_K = 32;
constexpr int TUNED_THREAD_TILE_M = 8;
constexpr int TUNED_THREAD_TILE_N = 4;
constexpr int TUNED_THREADS_X = TUNED_BLOCK_M / TUNED_THREAD_TILE_M;
constexpr int TUNED_THREADS_Y = TUNED_BLOCK_N / TUNED_THREAD_TILE_N;
constexpr int TUNED_THREADS = TUNED_THREADS_X * TUNED_THREADS_Y;

constexpr int SQUARE_BLOCK_M = 128;
constexpr int SQUARE_BLOCK_N = 128;
constexpr int SQUARE_BLOCK_K = 32;
constexpr int SQUARE_THREAD_TILE_M = 8;
constexpr int SQUARE_THREAD_TILE_N = 4;
constexpr int SQUARE_THREADS_X = SQUARE_BLOCK_M / SQUARE_THREAD_TILE_M;
constexpr int SQUARE_THREADS_Y = SQUARE_BLOCK_N / SQUARE_THREAD_TILE_N;
constexpr int SQUARE_THREADS = SQUARE_THREADS_X * SQUARE_THREADS_Y;

constexpr int WIDE128_BLOCK_M = 64;
constexpr int WIDE128_BLOCK_N = 128;
constexpr int WIDE128_BLOCK_K = 32;
constexpr int WIDE128_THREAD_TILE_M = 4;
constexpr int WIDE128_THREAD_TILE_N = 4;
constexpr int WIDE128_THREADS_X = WIDE128_BLOCK_M / WIDE128_THREAD_TILE_M;
constexpr int WIDE128_THREADS_Y = WIDE128_BLOCK_N / WIDE128_THREAD_TILE_N;
constexpr int WIDE128_THREADS = WIDE128_THREADS_X * WIDE128_THREADS_Y;

constexpr int SQUARE4_BLOCK_M = 128;
constexpr int SQUARE4_BLOCK_N = 128;
constexpr int SQUARE4_BLOCK_K = 32;
constexpr int SQUARE4_THREAD_TILE_M = 4;
constexpr int SQUARE4_THREAD_TILE_N = 4;
constexpr int SQUARE4_THREADS_X = SQUARE4_BLOCK_M / SQUARE4_THREAD_TILE_M;
constexpr int SQUARE4_THREADS_Y = SQUARE4_BLOCK_N / SQUARE4_THREAD_TILE_N;
constexpr int SQUARE4_THREADS = SQUARE4_THREADS_X * SQUARE4_THREADS_Y;

enum class HssNtKernel {
    Auto,
    Legacy64x64x16,
    Tuned128x64x32,
    Square128x128x32,
    Wide64x128x32,
    Square4x128x128x32
};

__host__ HssNtKernel hss_requested_nt_kernel()
{
    const char* value = std::getenv("HYGON_HSS_KERNEL");
    if (value == nullptr) {
        return HssNtKernel::Auto;
    }

    if (std::strcmp(value, "legacy") == 0 ||
        std::strcmp(value, "LEGACY") == 0 ||
        std::strcmp(value, "64x64") == 0) {
        return HssNtKernel::Legacy64x64x16;
    }

    if (std::strcmp(value, "tuned") == 0 ||
        std::strcmp(value, "TUNED") == 0 ||
        std::strcmp(value, "wide") == 0 ||
        std::strcmp(value, "128x64") == 0) {
        return HssNtKernel::Tuned128x64x32;
    }

    if (std::strcmp(value, "square") == 0 ||
        std::strcmp(value, "SQUARE") == 0 ||
        std::strcmp(value, "128x128") == 0) {
        return HssNtKernel::Square128x128x32;
    }

    if (std::strcmp(value, "wide128") == 0 ||
        std::strcmp(value, "WIDE128") == 0 ||
        std::strcmp(value, "64x128") == 0) {
        return HssNtKernel::Wide64x128x32;
    }

    if (std::strcmp(value, "square4") == 0 ||
        std::strcmp(value, "SQUARE4") == 0 ||
        std::strcmp(value, "128x128x4") == 0) {
        return HssNtKernel::Square4x128x128x32;
    }

    return HssNtKernel::Auto;
}

__host__ bool hss_fast_nt_path_enabled()
{
    const char* value = std::getenv("HYGON_HSS_FAST");
    if (value == nullptr) {
        return true;
    }

    return !(std::strcmp(value, "0") == 0 ||
             std::strcmp(value, "OFF") == 0 ||
             std::strcmp(value, "off") == 0 ||
             std::strcmp(value, "FALSE") == 0 ||
             std::strcmp(value, "false") == 0 ||
             std::strcmp(value, "NO") == 0 ||
             std::strcmp(value, "no") == 0);
}

__host__ bool hss_can_use_fast_nt_path(
    int m,
    int n,
    int k,
    int block_m,
    int block_n,
    int block_k,
    bool subtract_update)
{
    return subtract_update &&
           hss_fast_nt_path_enabled() &&
           (m % block_m) == 0 &&
           (n % block_n) == 0 &&
           (k % block_k) == 0;
}

__host__ HssNtKernel hss_select_nt_kernel(int m, int n, int k)
{
    HssNtKernel requested = hss_requested_nt_kernel();
    if (requested != HssNtKernel::Auto) {
        return requested;
    }

    (void)k;
    if (m < 4096 || n < 4096) {
        return HssNtKernel::Legacy64x64x16;
    }

    return HssNtKernel::Square128x128x32;
}

__host__ const char* hss_nt_kernel_name(HssNtKernel kernel)
{
    switch (kernel) {
    case HssNtKernel::Auto:
        return "custom_nt_auto";
    case HssNtKernel::Legacy64x64x16:
        return "custom_nt_legacy_64x64x16";
    case HssNtKernel::Tuned128x64x32:
        return "custom_nt_tuned_128x64x32_mix";
    case HssNtKernel::Square128x128x32:
        return "custom_nt_square_128x128x32_mix";
    case HssNtKernel::Wide64x128x32:
        return "custom_nt_wide_64x128x32_4x4_mix";
    case HssNtKernel::Square4x128x128x32:
        return "custom_nt_square4_128x128x32_4x4_mix";
    }

    return "custom_nt_auto";
}

__device__ __forceinline__ float hss_mixed_fma(__half a, __half b, float acc)
{
    return __builtin_fmaf(__half2float(a), __half2float(b), acc);
}

__device__ __forceinline__ __half load_op_a(
    const __half* __restrict__ A,
    int row,
    int col,
    int lda,
    bool transA)
{
    return transA ? A[col + static_cast<size_t>(row) * lda]
                  : A[row + static_cast<size_t>(col) * lda];
}

__device__ __forceinline__ __half load_op_b_half(
    const __half* __restrict__ B,
    int row,
    int col,
    int ldb,
    bool transB)
{
    return transB ? B[col + static_cast<size_t>(row) * ldb]
                  : B[row + static_cast<size_t>(col) * ldb];
}

__global__ void hss_gemm_hh_kernel(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const __half* __restrict__ A,
    int lda,
    const __half* __restrict__ B,
    int ldb,
    float beta,
    float* __restrict__ C,
    int ldc)
{
    int row = blockIdx.x * TILE + threadIdx.x;
    int col = blockIdx.y * TILE + threadIdx.y;

    __shared__ __half tile_a[TILE][TILE];
    __shared__ __half tile_b[TILE][TILE];

    float acc = 0.0f;

    for (int tile = 0; tile < k; tile += TILE) {
        int a_col = tile + threadIdx.y;
        int b_row = tile + threadIdx.x;

        tile_a[threadIdx.x][threadIdx.y] =
            (row < m && a_col < k) ? load_op_a(A, row, a_col, lda, transA)
                                   : __float2half(0.0f);
        tile_b[threadIdx.x][threadIdx.y] =
            (b_row < k && col < n) ? load_op_b_half(B, b_row, col, ldb, transB)
                                   : __float2half(0.0f);

        __syncthreads();

        for (int q = 0; q < TILE; ++q) {
            acc = hss_mixed_fma(tile_a[threadIdx.x][q], tile_b[q][threadIdx.y], acc);
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        size_t idx = row + static_cast<size_t>(col) * ldc;
        C[idx] = alpha * acc + beta * C[idx];
    }

    return;
}

#define DEFINE_HSS_NT_HH_KERNEL(NAME, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_TILE_M, THREAD_TILE_N, THREADS) \
__global__ __launch_bounds__(THREADS) void NAME(                                          \
    int m,                                                                                \
    int n,                                                                                \
    int k,                                                                                \
    float alpha,                                                                          \
    const __half* __restrict__ A,                                                         \
    int lda,                                                                              \
    const __half* __restrict__ B,                                                         \
    int ldb,                                                                              \
    float beta,                                                                           \
    float* __restrict__ C,                                                                \
    int ldc,                                                                              \
    bool subtract_update)                                                                 \
{                                                                                         \
    int local_row_base = threadIdx.x * (THREAD_TILE_M);                                   \
    int local_col_base = threadIdx.y * (THREAD_TILE_N);                                   \
    int global_row_base = blockIdx.x * (BLOCK_M) + local_row_base;                        \
    int global_col_base = blockIdx.y * (BLOCK_N) + local_col_base;                        \
                                                                                          \
    __shared__ __half tile_a[(BLOCK_K)][(BLOCK_M)];                                       \
    __shared__ __half tile_b[(BLOCK_K)][(BLOCK_N)];                                       \
                                                                                          \
    float acc[(THREAD_TILE_M)][(THREAD_TILE_N)] = {};                                     \
                                                                                          \
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;                              \
                                                                                          \
    for (int tile_k = 0; tile_k < k; tile_k += (BLOCK_K)) {                               \
        for (int idx = linear_tid;                                                        \
             idx < (BLOCK_K) * (BLOCK_M);                                                 \
             idx += (THREADS)) {                                                          \
            int kk = idx / (BLOCK_M);                                                     \
            int row = idx - kk * (BLOCK_M);                                               \
            int global_row = blockIdx.x * (BLOCK_M) + row;                                \
            int global_k = tile_k + kk;                                                   \
                                                                                          \
            tile_a[kk][row] =                                                             \
                (global_row < m && global_k < k)                                          \
                    ? A[global_row + static_cast<size_t>(global_k) * lda]                 \
                    : __float2half(0.0f);                                                 \
        }                                                                                 \
                                                                                          \
        for (int idx = linear_tid;                                                        \
             idx < (BLOCK_K) * (BLOCK_N);                                                 \
             idx += (THREADS)) {                                                          \
            int kk = idx / (BLOCK_N);                                                     \
            int col = idx - kk * (BLOCK_N);                                               \
            int global_col = blockIdx.y * (BLOCK_N) + col;                                \
            int global_k = tile_k + kk;                                                   \
                                                                                          \
            tile_b[kk][col] =                                                             \
                (global_k < k && global_col < n)                                          \
                    ? B[global_col + static_cast<size_t>(global_k) * ldb]                 \
                    : __float2half(0.0f);                                                 \
        }                                                                                 \
                                                                                          \
        __syncthreads();                                                                  \
                                                                                          \
        _Pragma("unroll")                                                                \
        for (int kk = 0; kk < (BLOCK_K); ++kk) {                                          \
            __half b0 = tile_b[kk][local_col_base + 0];                                   \
            __half b1 = tile_b[kk][local_col_base + 1];                                   \
            __half b2 = tile_b[kk][local_col_base + 2];                                   \
            __half b3 = tile_b[kk][local_col_base + 3];                                   \
                                                                                          \
            _Pragma("unroll")                                                            \
            for (int i = 0; i < (THREAD_TILE_M); ++i) {                                   \
                __half a_frag = tile_a[kk][local_row_base + i];                           \
                acc[i][0] = hss_mixed_fma(a_frag, b0, acc[i][0]);                         \
                acc[i][1] = hss_mixed_fma(a_frag, b1, acc[i][1]);                         \
                acc[i][2] = hss_mixed_fma(a_frag, b2, acc[i][2]);                         \
                acc[i][3] = hss_mixed_fma(a_frag, b3, acc[i][3]);                         \
            }                                                                             \
        }                                                                                 \
                                                                                          \
        __syncthreads();                                                                  \
    }                                                                                     \
                                                                                          \
    _Pragma("unroll")                                                                    \
    for (int j = 0; j < (THREAD_TILE_N); ++j) {                                           \
        int col = global_col_base + j;                                                    \
        if (col >= n) {                                                                   \
            continue;                                                                     \
        }                                                                                 \
                                                                                          \
        _Pragma("unroll")                                                                \
        for (int i = 0; i < (THREAD_TILE_M); ++i) {                                       \
            int row = global_row_base + i;                                                \
            if (row < m) {                                                                \
                size_t c_idx = row + static_cast<size_t>(col) * ldc;                      \
                if (subtract_update) {                                                    \
                    C[c_idx] -= acc[i][j];                                                \
                } else {                                                                  \
                    C[c_idx] = alpha * acc[i][j] + beta * C[c_idx];                       \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
    }                                                                                     \
                                                                                          \
    return;                                                                               \
}

#define DEFINE_HSS_NT_HH_FAST_KERNEL(NAME, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_TILE_M, THREAD_TILE_N, THREADS) \
__global__ __launch_bounds__(THREADS) void NAME(                                               \
    int m,                                                                                     \
    int n,                                                                                     \
    int k,                                                                                     \
    const __half* __restrict__ A,                                                              \
    int lda,                                                                                   \
    const __half* __restrict__ B,                                                              \
    int ldb,                                                                                   \
    float* __restrict__ C,                                                                     \
    int ldc)                                                                                   \
{                                                                                              \
    (void)m;                                                                                   \
    (void)n;                                                                                   \
    int local_row_base = threadIdx.x * (THREAD_TILE_M);                                        \
    int local_col_base = threadIdx.y * (THREAD_TILE_N);                                        \
    int global_row_base = blockIdx.x * (BLOCK_M) + local_row_base;                             \
    int global_col_base = blockIdx.y * (BLOCK_N) + local_col_base;                             \
                                                                                               \
    __shared__ __half tile_a[(BLOCK_K)][(BLOCK_M)];                                            \
    __shared__ __half tile_b[(BLOCK_K)][(BLOCK_N)];                                            \
                                                                                               \
    float acc[(THREAD_TILE_M)][(THREAD_TILE_N)] = {};                                          \
                                                                                               \
    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;                                   \
                                                                                               \
    for (int tile_k = 0; tile_k < k; tile_k += (BLOCK_K)) {                                    \
        for (int idx = linear_tid;                                                             \
             idx < (BLOCK_K) * (BLOCK_M);                                                      \
             idx += (THREADS)) {                                                               \
            int kk = idx / (BLOCK_M);                                                          \
            int row = idx - kk * (BLOCK_M);                                                    \
            int global_row = blockIdx.x * (BLOCK_M) + row;                                     \
            int global_k = tile_k + kk;                                                        \
                                                                                               \
            tile_a[kk][row] = A[global_row + static_cast<size_t>(global_k) * lda];             \
        }                                                                                      \
                                                                                               \
        for (int idx = linear_tid;                                                             \
             idx < (BLOCK_K) * (BLOCK_N);                                                      \
             idx += (THREADS)) {                                                               \
            int kk = idx / (BLOCK_N);                                                          \
            int col = idx - kk * (BLOCK_N);                                                    \
            int global_col = blockIdx.y * (BLOCK_N) + col;                                     \
            int global_k = tile_k + kk;                                                        \
                                                                                               \
            tile_b[kk][col] = B[global_col + static_cast<size_t>(global_k) * ldb];             \
        }                                                                                      \
                                                                                               \
        __syncthreads();                                                                       \
                                                                                               \
        _Pragma("unroll")                                                                     \
        for (int kk = 0; kk < (BLOCK_K); ++kk) {                                               \
            __half b0 = tile_b[kk][local_col_base + 0];                                        \
            __half b1 = tile_b[kk][local_col_base + 1];                                        \
            __half b2 = tile_b[kk][local_col_base + 2];                                        \
            __half b3 = tile_b[kk][local_col_base + 3];                                        \
                                                                                               \
            _Pragma("unroll")                                                                 \
            for (int i = 0; i < (THREAD_TILE_M); ++i) {                                        \
                __half a_frag = tile_a[kk][local_row_base + i];                                \
                acc[i][0] = hss_mixed_fma(a_frag, b0, acc[i][0]);                              \
                acc[i][1] = hss_mixed_fma(a_frag, b1, acc[i][1]);                              \
                acc[i][2] = hss_mixed_fma(a_frag, b2, acc[i][2]);                              \
                acc[i][3] = hss_mixed_fma(a_frag, b3, acc[i][3]);                              \
            }                                                                                  \
        }                                                                                      \
                                                                                               \
        __syncthreads();                                                                       \
    }                                                                                          \
                                                                                               \
    _Pragma("unroll")                                                                         \
    for (int j = 0; j < (THREAD_TILE_N); ++j) {                                                \
        int col = global_col_base + j;                                                         \
        _Pragma("unroll")                                                                     \
        for (int i = 0; i < (THREAD_TILE_M); ++i) {                                            \
            int row = global_row_base + i;                                                     \
            size_t c_idx = row + static_cast<size_t>(col) * ldc;                               \
            C[c_idx] -= acc[i][j];                                                             \
        }                                                                                      \
    }                                                                                          \
                                                                                               \
    return;                                                                                    \
}

DEFINE_HSS_NT_HH_KERNEL(
    hss_gemm_nt_hh_kernel,
    NT_BLOCK_M,
    NT_BLOCK_N,
    NT_BLOCK_K,
    NT_THREAD_TILE_M,
    NT_THREAD_TILE_N,
    NT_THREADS)

DEFINE_HSS_NT_HH_KERNEL(
    hss_gemm_nt_tuned_hh_kernel,
    TUNED_BLOCK_M,
    TUNED_BLOCK_N,
    TUNED_BLOCK_K,
    TUNED_THREAD_TILE_M,
    TUNED_THREAD_TILE_N,
    TUNED_THREADS)

DEFINE_HSS_NT_HH_KERNEL(
    hss_gemm_nt_square_hh_kernel,
    SQUARE_BLOCK_M,
    SQUARE_BLOCK_N,
    SQUARE_BLOCK_K,
    SQUARE_THREAD_TILE_M,
    SQUARE_THREAD_TILE_N,
    SQUARE_THREADS)

DEFINE_HSS_NT_HH_KERNEL(
    hss_gemm_nt_wide128_hh_kernel,
    WIDE128_BLOCK_M,
    WIDE128_BLOCK_N,
    WIDE128_BLOCK_K,
    WIDE128_THREAD_TILE_M,
    WIDE128_THREAD_TILE_N,
    WIDE128_THREADS)

DEFINE_HSS_NT_HH_KERNEL(
    hss_gemm_nt_square4_hh_kernel,
    SQUARE4_BLOCK_M,
    SQUARE4_BLOCK_N,
    SQUARE4_BLOCK_K,
    SQUARE4_THREAD_TILE_M,
    SQUARE4_THREAD_TILE_N,
    SQUARE4_THREADS)

DEFINE_HSS_NT_HH_FAST_KERNEL(
    hss_gemm_nt_hh_fast_kernel,
    NT_BLOCK_M,
    NT_BLOCK_N,
    NT_BLOCK_K,
    NT_THREAD_TILE_M,
    NT_THREAD_TILE_N,
    NT_THREADS)

DEFINE_HSS_NT_HH_FAST_KERNEL(
    hss_gemm_nt_tuned_hh_fast_kernel,
    TUNED_BLOCK_M,
    TUNED_BLOCK_N,
    TUNED_BLOCK_K,
    TUNED_THREAD_TILE_M,
    TUNED_THREAD_TILE_N,
    TUNED_THREADS)

DEFINE_HSS_NT_HH_FAST_KERNEL(
    hss_gemm_nt_square_hh_fast_kernel,
    SQUARE_BLOCK_M,
    SQUARE_BLOCK_N,
    SQUARE_BLOCK_K,
    SQUARE_THREAD_TILE_M,
    SQUARE_THREAD_TILE_N,
    SQUARE_THREADS)

DEFINE_HSS_NT_HH_FAST_KERNEL(
    hss_gemm_nt_wide128_hh_fast_kernel,
    WIDE128_BLOCK_M,
    WIDE128_BLOCK_N,
    WIDE128_BLOCK_K,
    WIDE128_THREAD_TILE_M,
    WIDE128_THREAD_TILE_N,
    WIDE128_THREADS)

DEFINE_HSS_NT_HH_FAST_KERNEL(
    hss_gemm_nt_square4_hh_fast_kernel,
    SQUARE4_BLOCK_M,
    SQUARE4_BLOCK_N,
    SQUARE4_BLOCK_K,
    SQUARE4_THREAD_TILE_M,
    SQUARE4_THREAD_TILE_N,
    SQUARE4_THREADS)

#undef DEFINE_HSS_NT_HH_KERNEL
#undef DEFINE_HSS_NT_HH_FAST_KERNEL

} // namespace

extern "C" {

const char* hygon_hss_custom_kernel_name()
{
    return hss_nt_kernel_name(hss_requested_nt_kernel());
}

void launch_hss_gemm_true_kernel(
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const __half* A_half,
    int lda,
    const __half* B_half,
    int ldb,
    float beta,
    float* C_f32,
    int ldc,
    hipStream_t stream)
{
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) {
        return;
    }

    bool use_trans_a = (transA != rocblas_operation_none);
    bool use_trans_b = (transB != rocblas_operation_none);
    bool subtract_update = (alpha == -1.0f && beta == 1.0f);

    if (!use_trans_a && use_trans_b) {
        HssNtKernel kernel = hss_select_nt_kernel(m, n, k);
        if (kernel == HssNtKernel::Legacy64x64x16) {
            dim3 block(NT_THREADS_X, NT_THREADS_Y);
            dim3 grid(
                (m + NT_BLOCK_M - 1) / NT_BLOCK_M,
                (n + NT_BLOCK_N - 1) / NT_BLOCK_N);

            if (hss_can_use_fast_nt_path(
                    m, n, k, NT_BLOCK_M, NT_BLOCK_N, NT_BLOCK_K, subtract_update)) {
                hipLaunchKernelGGL(
                    hss_gemm_nt_hh_fast_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    C_f32,
                    ldc);
            } else {
                hipLaunchKernelGGL(
                    hss_gemm_nt_hh_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    alpha,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    beta,
                    C_f32,
                    ldc,
                    subtract_update);
            }
        } else if (kernel == HssNtKernel::Square128x128x32) {
            dim3 block(SQUARE_THREADS_X, SQUARE_THREADS_Y);
            dim3 grid(
                (m + SQUARE_BLOCK_M - 1) / SQUARE_BLOCK_M,
                (n + SQUARE_BLOCK_N - 1) / SQUARE_BLOCK_N);

            if (hss_can_use_fast_nt_path(
                    m, n, k, SQUARE_BLOCK_M, SQUARE_BLOCK_N, SQUARE_BLOCK_K, subtract_update)) {
                hipLaunchKernelGGL(
                    hss_gemm_nt_square_hh_fast_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    C_f32,
                    ldc);
            } else {
                hipLaunchKernelGGL(
                    hss_gemm_nt_square_hh_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    alpha,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    beta,
                    C_f32,
                    ldc,
                    subtract_update);
            }
        } else if (kernel == HssNtKernel::Wide64x128x32) {
            dim3 block(WIDE128_THREADS_X, WIDE128_THREADS_Y);
            dim3 grid(
                (m + WIDE128_BLOCK_M - 1) / WIDE128_BLOCK_M,
                (n + WIDE128_BLOCK_N - 1) / WIDE128_BLOCK_N);

            if (hss_can_use_fast_nt_path(
                    m, n, k, WIDE128_BLOCK_M, WIDE128_BLOCK_N, WIDE128_BLOCK_K, subtract_update)) {
                hipLaunchKernelGGL(
                    hss_gemm_nt_wide128_hh_fast_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    C_f32,
                    ldc);
            } else {
                hipLaunchKernelGGL(
                    hss_gemm_nt_wide128_hh_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    alpha,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    beta,
                    C_f32,
                    ldc,
                    subtract_update);
            }
        } else if (kernel == HssNtKernel::Square4x128x128x32) {
            dim3 block(SQUARE4_THREADS_X, SQUARE4_THREADS_Y);
            dim3 grid(
                (m + SQUARE4_BLOCK_M - 1) / SQUARE4_BLOCK_M,
                (n + SQUARE4_BLOCK_N - 1) / SQUARE4_BLOCK_N);

            if (hss_can_use_fast_nt_path(
                    m, n, k, SQUARE4_BLOCK_M, SQUARE4_BLOCK_N, SQUARE4_BLOCK_K, subtract_update)) {
                hipLaunchKernelGGL(
                    hss_gemm_nt_square4_hh_fast_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    C_f32,
                    ldc);
            } else {
                hipLaunchKernelGGL(
                    hss_gemm_nt_square4_hh_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    alpha,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    beta,
                    C_f32,
                    ldc,
                    subtract_update);
            }
        } else {
            dim3 block(TUNED_THREADS_X, TUNED_THREADS_Y);
            dim3 grid(
                (m + TUNED_BLOCK_M - 1) / TUNED_BLOCK_M,
                (n + TUNED_BLOCK_N - 1) / TUNED_BLOCK_N);

            if (hss_can_use_fast_nt_path(
                    m, n, k, TUNED_BLOCK_M, TUNED_BLOCK_N, TUNED_BLOCK_K, subtract_update)) {
                hipLaunchKernelGGL(
                    hss_gemm_nt_tuned_hh_fast_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    C_f32,
                    ldc);
            } else {
                hipLaunchKernelGGL(
                    hss_gemm_nt_tuned_hh_kernel,
                    grid,
                    block,
                    0,
                    stream,
                    m,
                    n,
                    k,
                    alpha,
                    A_half,
                    lda,
                    B_half,
                    ldb,
                    beta,
                    C_f32,
                    ldc,
                    subtract_update);
            }
        }
    } else {
        dim3 block(TILE, TILE);
        dim3 grid(
            (m + TILE - 1) / TILE,
            (n + TILE - 1) / TILE);

        hipLaunchKernelGGL(
            hss_gemm_hh_kernel,
            grid,
            block,
            0,
            stream,
            use_trans_a,
            use_trans_b,
            m,
            n,
            k,
            alpha,
            A_half,
            lda,
            B_half,
            ldb,
            beta,
            C_f32,
            ldc);
    }

    return;
}

} // extern "C"
