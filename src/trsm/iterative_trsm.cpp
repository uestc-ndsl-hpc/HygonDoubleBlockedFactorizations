#include "trsm/iterative_trsm.h"
#include "utils/common/error_check.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>

extern "C" {

void iterative_trsm_right_looking(
    rocblas_handle handle,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    if (n <= 0 || m <= 0) return;

    float one = 1.0f;
    float neg_one = -1.0f;

    // Right-TRSM: split m (triangular dimension is on RIGHT)
    for (int i = 0; i < m; i += nb) {

        int current_nb = (m - i < nb) ? (m - i) : nb;

        // ======================================================
        // 1. Solve panel:
        //    X_i * L_ii^T = B_i
        // ======================================================
        rocblas_strsm(
            handle,
            rocblas_side_right,              // ✔ FIX: right side
            rocblas_fill_lower,
            rocblas_operation_transpose,     // ✔ FIX: match L^T
            rocblas_diagonal_non_unit,
            n, current_nb, &one,
            A + (size_t)i * lda + i, lda,
            B + (size_t)i * ldb, ldb);

        // ======================================================
        // 2. Update trailing matrix:
        //    B_j -= X_i * L_ji^T
        // ======================================================
        if (i + current_nb < m) {

            int remaining = m - (i + current_nb);

            rocblas_sgemm(
                handle,
                rocblas_operation_none,
                rocblas_operation_transpose,  // ✔ FIX: L^T consistency
                n, remaining, current_nb,
                &neg_one,

                B + (size_t)i * ldb, ldb,     // X_i
                A + (size_t)i * lda + (i + current_nb), lda, // L_ji

                &one,
                B + (size_t)(i + current_nb) * ldb, ldb);
        }
    }

    return;
}

void streamed_trsm_right_looking(
    rocblas_handle handle,
    StreamPool* pool,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    if (n <= 0 || m <= 0) return;
    if (pool == nullptr || pool->size() < 2) {
        iterative_trsm_right_looking(handle, A, lda, B, ldb, n, m, nb);
        return;
    }

    float one = 1.0f;
    float neg_one = -1.0f;

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    auto& panel_update_ctx = pool->get(0);
    auto& trailing_update_ctx = pool->get(1);

    std::vector<hipEvent_t> events;
    events.reserve(static_cast<size_t>((m + nb - 1) / nb) * 3);

    hipEvent_t panel_ready = nullptr;
    hipEvent_t trailing_done = nullptr;

    auto create_event = [&]() {
        hipEvent_t event = nullptr;
        HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        events.push_back(event);
        return event;
    };

    for (int i = 0; i < m; i += nb) {
        int current_nb = std::min(nb, m - i);

        if (panel_ready != nullptr) {
            HIP_CHECK(hipStreamWaitEvent(main_stream, panel_ready, 0));
        }

        ROCBLAS_CHECK(rocblas_strsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            n, current_nb, &one,
            A + (size_t)i * lda + i, lda,
            B + (size_t)i * ldb, ldb));

        int next_start = i + current_nb;
        if (next_start >= m) {
            break;
        }

        hipEvent_t solve_done = create_event();
        HIP_CHECK(hipEventRecord(solve_done, main_stream));

        int next_nb = std::min(nb, m - next_start);
        int trailing_start = next_start + next_nb;
        int trailing_cols = m - trailing_start;

        if (trailing_done != nullptr) {
            HIP_CHECK(hipStreamWaitEvent(panel_update_ctx.stream, trailing_done, 0));
            HIP_CHECK(hipStreamWaitEvent(trailing_update_ctx.stream, trailing_done, 0));
        }

        HIP_CHECK(hipStreamWaitEvent(panel_update_ctx.stream, solve_done, 0));
        ROCBLAS_CHECK(rocblas_sgemm(
            panel_update_ctx.handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            n, next_nb, current_nb,
            &neg_one,
            B + (size_t)i * ldb, ldb,
            A + (size_t)i * lda + next_start, lda,
            &one,
            B + (size_t)next_start * ldb, ldb));

        panel_ready = create_event();
        HIP_CHECK(hipEventRecord(panel_ready, panel_update_ctx.stream));

        if (trailing_cols > 0) {
            HIP_CHECK(hipStreamWaitEvent(trailing_update_ctx.stream, solve_done, 0));
            ROCBLAS_CHECK(rocblas_sgemm(
                trailing_update_ctx.handle,
                rocblas_operation_none,
                rocblas_operation_transpose,
                n, trailing_cols, current_nb,
                &neg_one,
                B + (size_t)i * ldb, ldb,
                A + (size_t)i * lda + trailing_start, lda,
                &one,
                B + (size_t)trailing_start * ldb, ldb));

            trailing_done = create_event();
            HIP_CHECK(hipEventRecord(trailing_done, trailing_update_ctx.stream));
        } else {
            trailing_done = panel_ready;
        }
    }

    if (panel_ready != nullptr) {
        HIP_CHECK(hipStreamWaitEvent(main_stream, panel_ready, 0));
    }
    if (trailing_done != nullptr) {
        HIP_CHECK(hipStreamWaitEvent(main_stream, trailing_done, 0));
    }

    for (hipEvent_t event : events) {
        HIP_CHECK(hipEventDestroy(event));
    }

    return;
}

}
