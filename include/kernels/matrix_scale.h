#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 在对角线上寻找绝对值最大元素 (针对 SPD 矩阵)
 */
void launch_max_abs(
    const float* A,
    float* global_max,
    int n,
    int lda,
    hipStream_t stream);

/**
 * @brief 缩放下三角矩阵 (列主序合并访存优化版)
 */
void launch_scale_matrix(
    float* A,
    int rows,
    int cols,
    int lda,
    float scale,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif