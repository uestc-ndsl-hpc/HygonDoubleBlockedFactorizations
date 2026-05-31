#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "kernels/fp16_convert.h"
#include "utils/common/error_check.h"

extern "C" {

// ======================================================
// FP32 -> FP16 (合并内存访问)
// ======================================================
__global__ void fp32_to_fp16_kernel_coalesced(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst)
{
    int col = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols && row < rows)
    {
        size_t src_idx = row + (size_t)col * ld_src;
        size_t dst_idx = row + (size_t)col * ld_dst;

        dst[dst_idx] = __float2half(src[src_idx]);
    }
}

// ======================================================
// FP16 -> FP32 (合并内存访问)
// ======================================================
__global__ void fp16_to_fp32_kernel_coalesced(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst)
{
    int col = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols && row < rows)
    {
        size_t src_idx = row + (size_t)col * ld_src;
        size_t dst_idx = row + (size_t)col * ld_dst;

        dst[dst_idx] = __half2float(src[src_idx]);
    }
}

// ======================================================
// Launchers
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
    if (rows <= 0 || cols <= 0) return;

    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE, 1);

    dim3 grid(
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
        cols);

    hipLaunchKernelGGL(
        fp32_to_fp16_kernel_coalesced,
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

void launch_fp16_to_fp32(
    const __half* src,
    float* dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst,
    hipStream_t stream)
{
    if (rows <= 0 || cols <= 0) return;

    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE, 1);
    
    dim3 grid(
        (rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
        cols);

    hipLaunchKernelGGL(
        fp16_to_fp32_kernel_coalesced,
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