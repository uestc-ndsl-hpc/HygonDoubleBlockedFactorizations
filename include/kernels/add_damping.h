#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 给对角线元素增加阻尼 (A = A + lambda * I)
 */
void launch_add_damping(
    float* A,
    int n,
    int lda,
    float lambda,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif