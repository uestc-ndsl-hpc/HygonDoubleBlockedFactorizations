#include "kernels/hss_gemm.h"

#include <cstdlib>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#if defined(HYGON_ENABLE_ROCWMMA)
#include <rocwmma/rocwmma.hpp>
#endif

namespace {

constexpr int TILE = 16;
constexpr int NT_BLOCK_M = 64;
constexpr int NT_BLOCK_N = 64;
constexpr int NT_BLOCK_K = 16;
constexpr int NT_THREAD_TILE_M = 4;
constexpr int NT_THREAD_TILE_N = 4;

#if defined(HYGON_ENABLE_ROCWMMA)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WMMA_WARPS_M = 4;
constexpr int WMMA_WARPS_N = 4;
constexpr int WMMA_BLOCK_M = WMMA_M * WMMA_WARPS_M;
constexpr int WMMA_BLOCK_N = WMMA_N * WMMA_WARPS_N;
constexpr int WMMA_BLOCK_K = WMMA_K;
constexpr int WMMA_WAVE_SIZE = 64;
constexpr int WMMA_THREADS = WMMA_WARPS_M * WMMA_WARPS_N * WMMA_WAVE_SIZE;
#endif

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

__device__ __forceinline__ float load_op_b(
    const float* __restrict__ B,
    int row,
    int col,
    int ldb,
    bool transB)
{
    return transB ? B[col + static_cast<size_t>(row) * ldb]
                  : B[row + static_cast<size_t>(col) * ldb];
}

__global__ void hss_gemm_kernel(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    float alpha,
    const __half* __restrict__ A,
    int lda,
    const float* __restrict__ B,
    int ldb,
    float beta,
    float* __restrict__ C,
    int ldc)
{
    int row = blockIdx.x * TILE + threadIdx.x;
    int col = blockIdx.y * TILE + threadIdx.y;

    __shared__ __half tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    float acc = 0.0f;

    for (int tile = 0; tile < k; tile += TILE) {
        int a_col = tile + threadIdx.y;
        int b_row = tile + threadIdx.x;

        tile_a[threadIdx.x][threadIdx.y] =
            (row < m && a_col < k) ? load_op_a(A, row, a_col, lda, transA)
                                   : __float2half(0.0f);
        tile_b[threadIdx.x][threadIdx.y] =
            (b_row < k && col < n) ? load_op_b(B, b_row, col, ldb, transB)
                                   : 0.0f;

        __syncthreads();

        for (int q = 0; q < TILE; ++q) {
            acc += __half2float(tile_a[threadIdx.x][q]) * tile_b[q][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        size_t idx = row + static_cast<size_t>(col) * ldc;
        C[idx] = alpha * acc + beta * C[idx];
    }

    return;
}

__global__ void hss_gemm_nt_kernel(
    int m,
    int n,
    int k,
    float alpha,
    const __half* __restrict__ A,
    int lda,
    const float* __restrict__ B,
    int ldb,
    float beta,
    float* __restrict__ C,
    int ldc,
    bool subtract_update)
{
    int local_row_base = threadIdx.x * NT_THREAD_TILE_M;
    int local_col_base = threadIdx.y * NT_THREAD_TILE_N;
    int global_row_base = blockIdx.x * NT_BLOCK_M + local_row_base;
    int global_col_base = blockIdx.y * NT_BLOCK_N + local_col_base;

    __shared__ __half tile_a[NT_BLOCK_K][NT_BLOCK_M];
    __shared__ float tile_b[NT_BLOCK_K][NT_BLOCK_N];

    float acc[NT_THREAD_TILE_M][NT_THREAD_TILE_N] = {};

    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int tile_k = 0; tile_k < k; tile_k += NT_BLOCK_K) {
        for (int idx = linear_tid; idx < NT_BLOCK_K * NT_BLOCK_M; idx += threads_per_block) {
            int kk = idx / NT_BLOCK_M;
            int row = idx - kk * NT_BLOCK_M;
            int global_row = blockIdx.x * NT_BLOCK_M + row;
            int global_k = tile_k + kk;

            tile_a[kk][row] =
                (global_row < m && global_k < k)
                    ? A[global_row + static_cast<size_t>(global_k) * lda]
                    : __float2half(0.0f);
        }

        for (int idx = linear_tid; idx < NT_BLOCK_K * NT_BLOCK_N; idx += threads_per_block) {
            int kk = idx / NT_BLOCK_N;
            int col = idx - kk * NT_BLOCK_N;
            int global_col = blockIdx.y * NT_BLOCK_N + col;
            int global_k = tile_k + kk;

            tile_b[kk][col] =
                (global_k < k && global_col < n)
                    ? B[global_col + static_cast<size_t>(global_k) * ldb]
                    : 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < NT_BLOCK_K; ++kk) {
            float a_frag[NT_THREAD_TILE_M];
            float b_frag[NT_THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < NT_THREAD_TILE_M; ++i) {
                a_frag[i] = __half2float(tile_a[kk][local_row_base + i]);
            }

            #pragma unroll
            for (int j = 0; j < NT_THREAD_TILE_N; ++j) {
                b_frag[j] = tile_b[kk][local_col_base + j];
            }

            #pragma unroll
            for (int i = 0; i < NT_THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < NT_THREAD_TILE_N; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NT_THREAD_TILE_M; ++i) {
        int row = global_row_base + i;
        if (row >= m) {
            continue;
        }

        #pragma unroll
        for (int j = 0; j < NT_THREAD_TILE_N; ++j) {
            int col = global_col_base + j;
            if (col < n) {
                size_t c_idx = row + static_cast<size_t>(col) * ldc;
                if (subtract_update) {
                    C[c_idx] -= acc[i][j];
                } else {
                    C[c_idx] = alpha * acc[i][j] + beta * C[c_idx];
                }
            }
        }
    }

    return;
}

__global__ void hss_gemm_nt_f32a_kernel(
    int m,
    int n,
    int k,
    float alpha,
    const float* __restrict__ A,
    int lda,
    const float* __restrict__ B,
    int ldb,
    float beta,
    float* __restrict__ C,
    int ldc,
    bool subtract_update)
{
    int local_row_base = threadIdx.x * NT_THREAD_TILE_M;
    int local_col_base = threadIdx.y * NT_THREAD_TILE_N;
    int global_row_base = blockIdx.x * NT_BLOCK_M + local_row_base;
    int global_col_base = blockIdx.y * NT_BLOCK_N + local_col_base;

    __shared__ __half tile_a[NT_BLOCK_K][NT_BLOCK_M];
    __shared__ float tile_b[NT_BLOCK_K][NT_BLOCK_N];

    float acc[NT_THREAD_TILE_M][NT_THREAD_TILE_N] = {};

    int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;

    for (int tile_k = 0; tile_k < k; tile_k += NT_BLOCK_K) {
        for (int idx = linear_tid; idx < NT_BLOCK_K * NT_BLOCK_M; idx += threads_per_block) {
            int kk = idx / NT_BLOCK_M;
            int row = idx - kk * NT_BLOCK_M;
            int global_row = blockIdx.x * NT_BLOCK_M + row;
            int global_k = tile_k + kk;

            tile_a[kk][row] =
                (global_row < m && global_k < k)
                    ? __float2half(A[global_row + static_cast<size_t>(global_k) * lda])
                    : __float2half(0.0f);
        }

        for (int idx = linear_tid; idx < NT_BLOCK_K * NT_BLOCK_N; idx += threads_per_block) {
            int kk = idx / NT_BLOCK_N;
            int col = idx - kk * NT_BLOCK_N;
            int global_col = blockIdx.y * NT_BLOCK_N + col;
            int global_k = tile_k + kk;

            tile_b[kk][col] =
                (global_k < k && global_col < n)
                    ? B[global_col + static_cast<size_t>(global_k) * ldb]
                    : 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < NT_BLOCK_K; ++kk) {
            float a_frag[NT_THREAD_TILE_M];
            float b_frag[NT_THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < NT_THREAD_TILE_M; ++i) {
                a_frag[i] = __half2float(tile_a[kk][local_row_base + i]);
            }

            #pragma unroll
            for (int j = 0; j < NT_THREAD_TILE_N; ++j) {
                b_frag[j] = tile_b[kk][local_col_base + j];
            }

            #pragma unroll
            for (int i = 0; i < NT_THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < NT_THREAD_TILE_N; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < NT_THREAD_TILE_M; ++i) {
        int row = global_row_base + i;
        if (row >= m) {
            continue;
        }

        #pragma unroll
        for (int j = 0; j < NT_THREAD_TILE_N; ++j) {
            int col = global_col_base + j;
            if (col < n) {
                size_t c_idx = row + static_cast<size_t>(col) * ldc;
                if (subtract_update) {
                    C[c_idx] -= acc[i][j];
                } else {
                    C[c_idx] = alpha * acc[i][j] + beta * C[c_idx];
                }
            }
        }
    }

    return;
}

#if defined(HYGON_ENABLE_ROCWMMA)
__global__ __launch_bounds__(WMMA_THREADS) void hss_gemm_nt_f32a_wmma_kernel(
    int m,
    int n,
    int k,
    const float* __restrict__ A,
    int lda,
    const float* __restrict__ B,
    int ldb,
    float* __restrict__ C,
    int ldc)
{
    int linear_tid = static_cast<int>(threadIdx.x);
    int wave_id = linear_tid / WMMA_WAVE_SIZE;
    int wave_m = wave_id / WMMA_WARPS_N;
    int wave_n = wave_id - wave_m * WMMA_WARPS_N;

    int tile_row = blockIdx.x * WMMA_BLOCK_M + wave_m * WMMA_M;
    int tile_col = blockIdx.y * WMMA_BLOCK_N + wave_n * WMMA_N;

    __shared__ __half lds_a[WMMA_BLOCK_K][WMMA_BLOCK_M];
    __shared__ __half lds_b[WMMA_BLOCK_K][WMMA_BLOCK_N];
    __shared__ float lds_c[WMMA_BLOCK_N][WMMA_BLOCK_M];

    using frag_a_t = rocwmma::fragment<
        rocwmma::matrix_a,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        __half,
        rocwmma::col_major>;
    using frag_b_t = rocwmma::fragment<
        rocwmma::matrix_b,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        __half,
        rocwmma::row_major>;
    using frag_c_t = rocwmma::fragment<
        rocwmma::accumulator,
        WMMA_M,
        WMMA_N,
        WMMA_K,
        float>;

    frag_a_t frag_a;
    frag_b_t frag_b;
    frag_c_t frag_c;
    rocwmma::fill_fragment(frag_c, 0.0f);

    for (int tile_k = 0; tile_k < k; tile_k += WMMA_BLOCK_K) {
        for (int idx = linear_tid; idx < WMMA_BLOCK_K * WMMA_BLOCK_M; idx += WMMA_THREADS) {
            int kk = idx / WMMA_BLOCK_M;
            int row = idx - kk * WMMA_BLOCK_M;
            int global_row = blockIdx.x * WMMA_BLOCK_M + row;
            int global_k = tile_k + kk;

            lds_a[kk][row] =
                (global_row < m && global_k < k)
                    ? __float2half(A[global_row + static_cast<size_t>(global_k) * lda])
                    : __float2half(0.0f);
        }

        for (int idx = linear_tid; idx < WMMA_BLOCK_K * WMMA_BLOCK_N; idx += WMMA_THREADS) {
            int kk = idx / WMMA_BLOCK_N;
            int col = idx - kk * WMMA_BLOCK_N;
            int global_col = blockIdx.y * WMMA_BLOCK_N + col;
            int global_k = tile_k + kk;

            lds_b[kk][col] =
                (global_col < n && global_k < k)
                    ? __float2half(B[global_col + static_cast<size_t>(global_k) * ldb])
                    : __float2half(0.0f);
        }

        __syncthreads();

        const __half* tile_a = &lds_a[0][wave_m * WMMA_M];
        const __half* tile_b = &lds_b[0][wave_n * WMMA_N];
        rocwmma::load_matrix_sync(frag_a, tile_a, WMMA_BLOCK_M);
        rocwmma::load_matrix_sync(frag_b, tile_b, WMMA_BLOCK_N);
        rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

        __syncthreads();
    }

    float* tile_scratch = &lds_c[wave_n * WMMA_N][wave_m * WMMA_M];
    rocwmma::store_matrix_sync(tile_scratch, frag_c, WMMA_BLOCK_M, rocwmma::mem_col_major);
    __syncthreads();

    for (int idx = linear_tid; idx < WMMA_BLOCK_M * WMMA_BLOCK_N; idx += WMMA_THREADS) {
        int local_col = idx / WMMA_BLOCK_M;
        int local_row = idx - local_col * WMMA_BLOCK_M;
        int row = blockIdx.x * WMMA_BLOCK_M + local_row;
        int col = blockIdx.y * WMMA_BLOCK_N + local_col;
        if (row < m && col < n) {
            size_t c_idx = row + static_cast<size_t>(col) * ldc;
            C[c_idx] -= lds_c[local_col][local_row];
        }
    }
}
#endif

__host__ bool hss_wmma_disabled()
{
#if defined(HYGON_ENABLE_ROCWMMA)
    const char* value = std::getenv("HYGON_HSS_DISABLE_WMMA");
    if (value == nullptr) {
        return false;
    }

    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
#else
    return true;
#endif
}

} // namespace

extern "C" {

void launch_hss_gemm_kernel(
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const __half* A_half,
    int lda,
    const float* B_f32,
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
        dim3 block(NT_BLOCK_M / NT_THREAD_TILE_M, NT_BLOCK_N / NT_THREAD_TILE_N);
        dim3 grid(
            (m + NT_BLOCK_M - 1) / NT_BLOCK_M,
            (n + NT_BLOCK_N - 1) / NT_BLOCK_N);

        hipLaunchKernelGGL(
            hss_gemm_nt_kernel,
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
            B_f32,
            ldb,
            beta,
            C_f32,
            ldc,
            subtract_update);
    } else {
        dim3 block(TILE, TILE);
        dim3 grid(
            (m + TILE - 1) / TILE,
            (n + TILE - 1) / TILE);

        hipLaunchKernelGGL(
            hss_gemm_kernel,
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
            B_f32,
            ldb,
            beta,
            C_f32,
            ldc);
    }

    return;
}

void launch_hss_gemm_f32a_kernel(
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const float* A_f32,
    int lda,
    const float* B_f32,
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
#if defined(HYGON_ENABLE_ROCWMMA)
        if (subtract_update && !hss_wmma_disabled()) {
            dim3 block(WMMA_THREADS);
            dim3 grid(
                (m + WMMA_BLOCK_M - 1) / WMMA_BLOCK_M,
                (n + WMMA_BLOCK_N - 1) / WMMA_BLOCK_N);

            hipLaunchKernelGGL(
                hss_gemm_nt_f32a_wmma_kernel,
                grid,
                block,
                0,
                stream,
                m,
                n,
                k,
                A_f32,
                lda,
                B_f32,
                ldb,
                C_f32,
                ldc);
            return;
        }
#endif

        dim3 block(NT_BLOCK_M / NT_THREAD_TILE_M, NT_BLOCK_N / NT_THREAD_TILE_N);
        dim3 grid(
            (m + NT_BLOCK_M - 1) / NT_BLOCK_M,
            (n + NT_BLOCK_N - 1) / NT_BLOCK_N);

        hipLaunchKernelGGL(
            hss_gemm_nt_f32a_kernel,
            grid,
            block,
            0,
            stream,
            m,
            n,
            k,
            alpha,
            A_f32,
            lda,
            B_f32,
            ldb,
            beta,
            C_f32,
            ldc,
            subtract_update);
    }

    return;
}

} // extern "C"
