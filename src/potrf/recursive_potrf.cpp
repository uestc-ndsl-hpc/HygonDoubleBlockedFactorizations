#include "potrf/recursive_potrf.h"

#include "trsm/recursive_trsm.h"
#include "syrk/batched_syrk.h"
#include "utils/common/error_check.h"
#include "kernels/add_damping.h"
#include "kernels/matrix_scale.h"

#include <rocsolver/rocsolver.h>

extern "C" {

// ======================================================
// 单精度
// ======================================================

void recursive_potrf(rocblas_handle handle, StreamPool* pool,
                     float* A,
                     int lda,
                     int n,
                     int nb,
                     int* info)
{
    if (n <= 0)
        return;

    // ==================================================
    // Base case
    // ==================================================

    if (n <= nb)
    {
        rocsolver_spotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info);

        return;
    }

    // ==================================================
    // Split
    // ==================================================

    int n1 = n / 2;
    int n2 = n - n1;

    float* A11 = A;

    float* A21 = A + n1;

     float* A22 = A + (size_t)n1 * (size_t)lda + n1;

    // ==================================================
    // POTRF(A11)
    // ==================================================

    recursive_potrf(
        handle, pool,
        A11,
        lda,
        n1,
        nb,
        info);

    // ==================================================
    // TRSM
    // ==================================================

    recursive_trsm(
        handle, pool,
        A11, lda,
        A21, lda,
        n2, n1, nb);

    // ==================================================
    // SYRK
    // ==================================================

    batched_syrk_bottom_up(
        handle, pool,
        A21,
        lda,
        A22,
        lda,
        n2,
        n1,
        nb);

    // ==================================================
    // POTRF(A22)
    // ==================================================

    recursive_potrf(
        handle, pool,
        A22,
        lda,
        n2,
        nb,
        info);
}

// ======================================================
// 双精度
// ======================================================

void recursive_potrf_double(rocblas_handle handle, StreamPool* pool,
                            double* A,
                            int lda,
                            int n,
                            int nb,
                            int* info)
{
    if (n <= 0)
        return;

    if (n <= nb)
    {
        rocsolver_dpotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info);

        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    double* A11 = A;

    double* A21 = A + n1;

    double* A22 = A + (size_t)n1 * (size_t)lda + n1;

    recursive_potrf_double(
        handle, pool,
        A11,
        lda,
        n1,
        nb,
        info);

    recursive_trsm_double(
        handle, pool,
        A11, lda,
        A21, lda,
        n2, n1, nb);

    batched_syrk_bottom_up_double(
        handle, pool,
        A21,
        lda,
        A22,
        lda,
        n2,
        n1,
        nb);

    recursive_potrf_double(
        handle, pool,
        A22,
        lda,
        n2,
        nb,
        info);
}

void recursive_potrf_mixed(rocblas_handle handle, StreamPool* pool,
                           float* A,
                           int lda,
                           int n,
                           int nb,
                           int* info,
                           float lambda,
                           float* d_leaf_backup = nullptr)
{
    if (n <= 0)
        return;

    // --------------------------------------------------------------------------------------
    // Base Case
    // --------------------------------------------------------------------------------------
    if (n <= nb)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        float* A_backup = d_leaf_backup;
        bool use_temp_backup = (d_leaf_backup == nullptr);
        size_t backup_pitch = n * sizeof(float);

        if (use_temp_backup) {
            HIP_CHECK(hipMallocAsync(&A_backup, n * n * sizeof(float), stream));
        }

        // 保存原始备份
        HIP_CHECK(hipMemcpy2DAsync(
            A_backup, backup_pitch,
            A, lda * sizeof(float),
            n * sizeof(float), n,
            hipMemcpyDeviceToDevice, stream));

        // 第一次尝试不加阻尼分解
        rocsolver_spotrf(handle, rocblas_fill_lower, n, A, lda, info);

        int host_info = 0;
        HIP_CHECK(hipMemcpyAsync(&host_info, info, sizeof(int), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        // 如果检测到非正定，启动阻尼恢复循环
        if (host_info > 0)
        {
            int attempts = 0;
            float current_lambda = lambda;

            while (host_info > 0 && attempts < 5)
            {
                HIP_CHECK(hipMemcpy2DAsync(
                    A, lda * sizeof(float),
                    A_backup, backup_pitch,
                    n * sizeof(float), n,
                    hipMemcpyDeviceToDevice, stream));

                int zero = 0;
                HIP_CHECK(hipMemcpyAsync(info, &zero, sizeof(int), hipMemcpyHostToDevice, stream));

                launch_add_damping(A, n, lda, current_lambda, stream);

                rocsolver_spotrf(handle, rocblas_fill_lower, n, A, lda, info);

                HIP_CHECK(hipMemcpyAsync(&host_info, info, sizeof(int), hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                current_lambda *= 2.0f;
                attempts++;
            }
        }

        if (use_temp_backup) {
            HIP_CHECK(hipFreeAsync(A_backup, stream));
        }
        return;
    }

    // --------------------------------------------------------------------------------------
    // Split
    // --------------------------------------------------------------------------------------
    int n1 = n / 2;
    int n2 = n - n1;

    float* A11 = A;
    float* A21 = A + n1;
    float* A22 = A + (size_t)n1 * (size_t)lda + n1;

    // 1. POTRF(A11) - 分解左上角
    recursive_potrf_mixed(handle, pool, A11, lda, n1, nb, info, lambda, d_leaf_backup);

    // 2. TRSM - 通过 Tensor Cores 加速（全异步执行，不进行任何 CPU-GPU 握手同步）
    recursive_trsm_mixed(handle, pool, A11, lda, A21, lda, n2, n1, nb);

    // 3. SYRK - 通过 Tensor Cores 加速
    batched_syrk_bottom_up_mixed(handle, pool, A21, lda, A22, lda, n2, n1, nb);

    // 4. POTRF(A22) - 分解右下角
    recursive_potrf_mixed(handle, pool, A22, lda, n2, nb, info, lambda, d_leaf_backup);
}

// ==========================================================================================
// 统一入口控制驱动 (Driver)
// ==========================================================================================
void recursive_potrf_mixed_driver(
    rocblas_handle handle,
    StreamPool* pool,
    float* A,
    int lda,
    int n,
    int nb,
    int* info)
{
    if (n <= 0) return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // 1. 执行自动前向缩放（统一到最大 65504.0f）
    unsigned int* d_max_val = nullptr;
    HIP_CHECK(hipMallocAsync(&d_max_val, sizeof(unsigned int), stream));
    launch_auto_scale(A, n, lda, d_max_val, false, stream);
    
    // 2. 预分配叶子节点备份空间
    float* d_leaf_backup = nullptr;
    HIP_CHECK(hipMallocAsync(&d_leaf_backup, (size_t)nb * nb * sizeof(float), stream));

    // 3. 初始阻尼值计算
    float scaled_lambda = SCALE_TARGET * 1e-4f; 

    // 4. 提交全异步递归 Cholesky（GPU 队列中将实现无阻碍的流水线并行）
    recursive_potrf_mixed(handle, pool, A, lda, n, nb, info, scaled_lambda, d_leaf_backup);

    // 5. 还原缩放
    launch_auto_scale(A, n, lda, d_max_val, true, stream);

    // 6. 异步清理工作区
    HIP_CHECK(hipFreeAsync(d_leaf_backup, stream));
    HIP_CHECK(hipFreeAsync(d_max_val, stream));
}

} // extern "C"