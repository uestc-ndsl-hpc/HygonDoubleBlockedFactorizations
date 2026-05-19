#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "kernels/fp16_convert.h"

extern "C" {

// ======================================================
// FP32 -> FP16
// Column-major matrix conversion
// ======================================================

__global__ void fp32_to_fp16_kernel(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols)
    {
        int src_idx = row + col * ld_src;
        int dst_idx = row + col * ld_dst;

        dst[dst_idx] = __float2half(src[src_idx]);
    }
}

// ======================================================
// launcher
// ======================================================

void launch_fp32_to_fp16(
    const float* src,
    __half* dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst,
    hipStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid(
        (rows + block.x - 1) / block.x,
        (cols + block.y - 1) / block.y);

    hipLaunchKernelGGL(
        fp32_to_fp16_kernel,
        grid,
        block,
        0,
        stream,
        src,
        dst,
        rows,
        cols,
        ld_src,
        ld_dst);
}

} // extern "C"