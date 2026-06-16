#include "kernels/hss_gemm.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int TILE = 16;
constexpr int NT_BLOCK_M = 64;
constexpr int NT_BLOCK_N = 64;
constexpr int NT_BLOCK_K = 16;
constexpr int NT_THREAD_TILE_M = 4;
constexpr int NT_THREAD_TILE_N = 4;

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
