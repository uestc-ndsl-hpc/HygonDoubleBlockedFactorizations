#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" {

// ======================================================
// FP32 -> FP16
// Column-major matrix conversion
// ======================================================
void launch_fp32_to_fp16(
    const float* src,
    __half* dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst,
    hipStream_t stream);

// ======================================================
// FP16 -> FP32
// Column-major matrix conversion
// ======================================================
void launch_fp16_to_fp32(
    const __half* src,
    float* dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst,
    hipStream_t stream);

} // extern "C"