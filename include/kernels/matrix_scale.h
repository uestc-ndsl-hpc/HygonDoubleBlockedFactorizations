#pragma once
#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 全异步矩阵自动缩放接口（支持正向缩放与反向恢复）
 * 
 * @param A 矩阵 A 指针
 * @param n 矩阵阶数
 * @param lda 矩阵 A 的主维度
 * @param d_max_val 设备端临时空间，用于存储最大绝对值
 * @param inverse false 表示正向自适应缩放，true 表示逆向数据还原
 * @param stream 计算流
 */
void launch_auto_scale(
    float* A,
    int n,
    int lda,
    unsigned int* d_max_val,
    bool inverse,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif