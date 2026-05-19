#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 将 FP32 矩阵块转换为 FP16
 */
void launch_fp32_to_fp16(
    const float* src,
    __half* dst,
    int rows,
    int cols,
    int ld_src,
    int ld_dst,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif