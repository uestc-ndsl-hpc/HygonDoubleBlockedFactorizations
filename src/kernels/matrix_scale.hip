#include <hip/hip_runtime.h>
#include <cmath>
#include "kernels/matrix_scale.h"

extern "C" {

// ======================================================
// SPD scaling proxy: find max abs on diagonal
// ======================================================
__global__ void max_abs_kernel(
    const float* __restrict__ A,
    float* __restrict__ global_max,
    int n,
    int lda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // For SPD, diagonal elements are sufficient proxies for magnitude
        float val = fabsf(A[idx + idx * lda]);
        unsigned int iv = __float_as_uint(val);
        atomicMax((unsigned int*)global_max, iv);
    }
}

// ======================================================
// Scale lower triangle (Industry standard mapping)
// ======================================================
__global__ void scale_matrix_kernel(
    float* __restrict__ A,
    int rows,
    int cols,
    int lda,
    float scale)
{
    // Industry convention: 
    // x dimension maps to independent work units (columns)
    // y dimension maps to fast-changing index (rows)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check and lower triangle mask
    if (col < cols && row < rows && col <= row)
    {
        A[row + col * lda] *= scale;
    }
}

void launch_max_abs(
    const float* A,
    float* global_max,
    int n,
    int lda,
    hipStream_t stream)
{
    if (n <= 0) return;
    constexpr int BLOCK = 256;
    int GRID = (n + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(max_abs_kernel, dim3(GRID), dim3(BLOCK), 0, stream, A, global_max, n, lda);
}

void launch_scale_matrix(
    float* A,
    int rows,
    int cols,
    int lda,
    float scale,
    hipStream_t stream)
{
    // Industry Standard: x for cols, y for rows
    dim3 block(16, 16); 
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    hipLaunchKernelGGL(
        scale_matrix_kernel,
        grid,
        block,
        0,
        stream,
        A,
        rows,
        cols,
        lda,
        scale);
}

} // extern "C"