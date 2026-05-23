#include <hip/hip_runtime.h>
#include <cmath>
#include "kernels/matrix_scale.h"

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

// ======================================================
// 双向对角线自适应缩放（支持正向缩放和逆向还原）
// ======================================================
__global__ void scale_matrix_kernel(
    float* __restrict__ A,
    int rows,
    int cols,
    int lda,
    const unsigned int* __restrict__ global_max_uint_ptr,
    bool inverse) // false: 正向缩放, true: 逆向还原
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows && col <= row)
    {
        unsigned int max_uint = *global_max_uint_ptr;
        float max_val = __uint_as_float(max_uint);
        float scale = 1.0f;

        if (max_val > 0.0f)
        {
            if (inverse) {
                // 逆向还原：1 / (50000 / max_val) = max_val / 50000
                scale = max_val / 50000.0f;
            } else {
                // 正向自适应缩放
                scale = 50000.0f / max_val;
            }
        }
        A[row + col * lda] *= scale;
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
        // 1. 正向缩放流程：异步清零 -> 异步检索最大值
        hipLaunchKernelGGL(clear_max_kernel, dim3(1), dim3(1), 0, stream, d_max_val);

        constexpr int BLOCK = 256;
        int GRID = (n + BLOCK - 1) / BLOCK;
        hipLaunchKernelGGL(max_abs_kernel, dim3(GRID), dim3(BLOCK), 0, stream, A, d_max_val, n, lda);
    }

    // 2. 双向自适应缩放应用
    dim3 block(16, 16); 
    dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y
    );

    hipLaunchKernelGGL(
        scale_matrix_kernel,
        grid,
        block,
        0,
        stream,
        A,
        n,
        n,
        lda,
        d_max_val,
        inverse);
}

} // extern "C"