#include <hip/hip_runtime.h>
#include <cmath>

extern "C" {

// ======================================================
// 异步清零辅助核函数
// ======================================================
__global__ void clear_max_kernel(unsigned int* global_max)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *global_max = 0;
    }
}

// ======================================================
// SPD 缩放代理：在对角线上寻找绝对值最大值
// ======================================================
__global__ void max_abs_kernel(
    const float* __restrict__ A,
    unsigned int* __restrict__ global_max,
    int n,
    int lda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = fabsf(A[idx + (size_t)idx * lda]);
        unsigned int iv = __float_as_uint(val);
        atomicMax(global_max, iv);
    }
}

__global__ void scale_matrix_kernel_coalesced(
    float* __restrict__ A,
    int n,
    int lda,
    const unsigned int* __restrict__ global_max_uint_ptr,
    bool inverse) // false: 正向缩放, true: Cholesky 逆向还原
{
    int col = blockIdx.x;

    __shared__ float scale;
    if (threadIdx.x == 0)
    {
        unsigned int max_uint = *global_max_uint_ptr;
        float max_val = __uint_as_float(max_uint);
        if (max_val > 0.0f) {
            if (inverse) {
                scale = sqrtf(max_val / 50000.0f);
            } else {
                scale = 50000.0f / max_val;
            }
        } else {
            scale = 1.0f;
        }
    }
    __syncthreads();

    int start_row = col;
    int end_row = n;

    for (int r = start_row + threadIdx.x; r < end_row; r += blockDim.x)
    {
        A[r + (size_t)col * lda] *= scale;
    }
}

// ======================================================
// 全异步统一控制 Launcher
// ======================================================
void launch_auto_scale(
    float* A,
    int n,
    int lda,
    unsigned int* d_max_val,
    bool inverse,
    hipStream_t stream)
{
    if (n <= 0) return;

    if (!inverse)
    {
        hipLaunchKernelGGL(clear_max_kernel, dim3(1), dim3(1), 0, stream, d_max_val);

        constexpr int BLOCK = 256;
        int GRID = (n + BLOCK - 1) / BLOCK;
        hipLaunchKernelGGL(max_abs_kernel, dim3(GRID), dim3(BLOCK), 0, stream, A, d_max_val, n, lda);
    }

    constexpr int BLOCK_SIZE = 256;
    hipLaunchKernelGGL(
        scale_matrix_kernel_coalesced,
        dim3(n),            // 启动 n 个线程块，每个线程块处理一列
        dim3(BLOCK_SIZE),   // 每个线程块包含 256 个线程
        0,
        stream,
        A,
        n,
        lda,
        d_max_val,
        inverse);
}

} // extern "C"