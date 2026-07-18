#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "getrf/double_blocked_getrf.h"
#include "getrf/getrf_pivot_kernels.h"
#include "getrf_pivot_internal.h"
#include "utils/common/error_check.h"

namespace getrf_batched_pivot_internal {

struct NullObserver {
    template <typename Func>
    void measure(const char*, double, Func&& func)
    {
        func();

        return;
    }
};

inline double panel_getrf_work(int rows, int cols)
{
    const double m = static_cast<double>(rows);
    const double n = static_cast<double>(cols);
    return m * n * n - n * n * n / 3.0;
}

inline double trsm_work(int rows, int cols)
{
    return static_cast<double>(rows) * rows * cols;
}

inline double gemm_work(int rows, int cols, int k)
{
    return 2.0 * static_cast<double>(rows) * cols * k;
}

inline bool workspace_is_valid(
    const getrf_batched_pivot_workspace* workspace,
    int n)
{
    return workspace != nullptr
        && workspace->device_sources != nullptr
        && workspace->device_targets != nullptr
        && workspace->device_swap_buffer != nullptr
        && workspace->host_pivots != nullptr
        && workspace->host_permutation != nullptr
        && workspace->host_sources != nullptr
        && workspace->host_targets != nullptr
        && workspace->capacity >= n
        && workspace->column_tile > 0;
}

inline int collect_changed_rows(getrf_batched_pivot_workspace& workspace,
                                int row_count)
{
    int move_count = 0;
    for (int row = 0; row < row_count; ++row) {
        const int source = workspace.host_permutation[row];
        if (source == row) {
            continue;
        }

        workspace.host_sources[move_count] = source;
        workspace.host_targets[move_count] = row;
        ++move_count;
    }

    return move_count;
}

inline void upload_permutation(getrf_batched_pivot_workspace& workspace,
                               int move_count,
                               hipStream_t stream)
{
    if (move_count <= 0) {
        return;
    }

    const size_t bytes = static_cast<size_t>(move_count) * sizeof(int);
    HIP_CHECK(hipMemcpyAsync(
        workspace.device_sources,
        workspace.host_sources,
        bytes,
        hipMemcpyHostToDevice,
        stream));
    HIP_CHECK(hipMemcpyAsync(
        workspace.device_targets,
        workspace.host_targets,
        bytes,
        hipMemcpyHostToDevice,
        stream));

    return;
}

inline int prepare_panel_permutation(hipStream_t stream,
                                     int* device_pivots,
                                     int panel,
                                     int panel_size,
                                     int n,
                                     getrf_batched_pivot_workspace& workspace)
{
    const size_t bytes = static_cast<size_t>(panel_size) * sizeof(int);
    int* host_pivots = workspace.host_pivots + panel;

    HIP_CHECK(hipMemcpyAsync(
        host_pivots,
        device_pivots,
        bytes,
        hipMemcpyDeviceToHost,
        stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    const int active_rows = n - panel;
    for (int row = 0; row < active_rows; ++row) {
        workspace.host_permutation[row] = row;
    }

    for (int offset = 0; offset < panel_size; ++offset) {
        const int local_pivot = host_pivots[offset] - 1;
        if (local_pivot < offset || local_pivot >= active_rows) {
            return -1;
        }

        std::swap(
            workspace.host_permutation[offset],
            workspace.host_permutation[local_pivot]);
        host_pivots[offset] += panel;
    }

    HIP_CHECK(hipMemcpyAsync(
        device_pivots,
        host_pivots,
        bytes,
        hipMemcpyHostToDevice,
        stream));

    const int move_count = collect_changed_rows(workspace, active_rows);
    upload_permutation(workspace, move_count, stream);

    return move_count;
}

inline int prepare_history_permutation(
    hipStream_t stream,
    int first_pivot,
    int pivot_end,
    int row_base,
    int n,
    getrf_batched_pivot_workspace& workspace)
{
    // The host source/target arrays back asynchronous H2D copies from the
    // previous batch, so do not overwrite them until that stream is done.
    HIP_CHECK(hipStreamSynchronize(stream));

    const int row_count = n - row_base;
    for (int row = 0; row < row_count; ++row) {
        workspace.host_permutation[row] = row;
    }

    for (int row = first_pivot; row < pivot_end; ++row) {
        const int target = row - row_base;
        const int pivot = workspace.host_pivots[row] - 1 - row_base;
        if (target < 0 || target >= row_count
            || pivot < target || pivot >= row_count) {
            return -1;
        }

        std::swap(
            workspace.host_permutation[target],
            workspace.host_permutation[pivot]);
    }

    const int move_count = collect_changed_rows(workspace, row_count);
    upload_permutation(workspace, move_count, stream);

    return move_count;
}

template <typename T>
void apply_uploaded_permutation(hipStream_t stream,
                                T* A,
                                int lda,
                                int column_count,
                                int move_count,
                                getrf_batched_pivot_workspace& workspace)
{
    if (column_count <= 0 || move_count <= 0) {
        return;
    }

    if constexpr (std::is_same_v<T, float>) {
        launch_apply_getrf_permutation(
            A,
            lda,
            column_count,
            workspace.device_sources,
            workspace.device_targets,
            move_count,
            static_cast<float*>(workspace.device_swap_buffer),
            workspace.column_tile,
            stream);
    } else {
        launch_apply_getrf_permutation_double(
            A,
            lda,
            column_count,
            workspace.device_sources,
            workspace.device_targets,
            move_count,
            static_cast<double*>(workspace.device_swap_buffer),
            workspace.column_tile,
            stream);
    }

    return;
}

template <typename T, typename Observer>
void trsm_left_lower_unit_iterative(rocblas_handle handle,
                                    Observer& observer,
                                    int rows,
                                    int cols,
                                    const T* L,
                                    int ldl,
                                    T* B,
                                    int ldb,
                                    int block_size)
{
    using getrf_pivot_internal::block_ptr;
    using getrf_pivot_internal::gemm_nn_update;
    using getrf_pivot_internal::trsm_left_lower_unit;

    if (rows <= 0 || cols <= 0) {
        return;
    }

    const int step = block_size > 0 ? block_size : rows;
    for (int row = 0; row < rows; row += step) {
        const int diagonal_rows = std::min(step, rows - row);
        observer.measure(
            "Outer TRSM block",
            trsm_work(diagonal_rows, cols),
            [&]() {
                trsm_left_lower_unit(
                    handle,
                    diagonal_rows,
                    cols,
                    block_ptr(L, ldl, row, row),
                    ldl,
                    B + row,
                    ldb);
            });

        const int trailing_rows = rows - row - diagonal_rows;
        if (trailing_rows > 0) {
            observer.measure(
                "Outer TRSM GEMM",
                gemm_work(trailing_rows, cols, diagonal_rows),
                [&]() {
                    gemm_nn_update(
                        handle,
                        trailing_rows,
                        cols,
                        diagonal_rows,
                        block_ptr(L, ldl, row + diagonal_rows, row),
                        ldl,
                        B + row,
                        ldb,
                        B + row + diagonal_rows,
                        ldb);
                });
        }
    }

    return;
}

template <typename T, typename Observer>
void gemm_nn_update_m_tiled(rocblas_handle handle,
                            Observer& observer,
                            const char* stage_name,
                            int rows,
                            int cols,
                            int k,
                            const T* A,
                            int lda,
                            const T* B,
                            int ldb,
                            T* C,
                            int ldc,
                            int m_tile)
{
    using getrf_pivot_internal::gemm_nn_update;

    const int tile = m_tile > 0 ? std::min(m_tile, rows) : rows;
    observer.measure(
        stage_name,
        gemm_work(rows, cols, k),
        [&]() {
            for (int row = 0; row < rows; row += tile) {
                const int tile_rows = std::min(tile, rows - row);
                gemm_nn_update(
                    handle,
                    tile_rows,
                    cols,
                    k,
                    A + row,
                    lda,
                    B,
                    ldb,
                    C + row,
                    ldc);
            }
        });

    return;
}

template <typename T, typename Observer>
void prepare_next_inner_panel(rocblas_handle handle,
                              Observer& observer,
                              T* A,
                              int lda,
                              int n,
                              int outer,
                              int next_inner,
                              int outer_end,
                              int inner_nb,
                              int inner_gemm_m_tile)
{
    using getrf_pivot_internal::block_ptr;
    using getrf_pivot_internal::trsm_left_lower_unit;

    if (next_inner >= outer_end) {
        return;
    }

    const int next_size = std::min(inner_nb, outer_end - next_inner);
    const int history_size = next_inner - outer;

    observer.measure(
        "Inner TRSM",
        trsm_work(history_size, next_size),
        [&]() {
            trsm_left_lower_unit(
                handle,
                history_size,
                next_size,
                block_ptr(A, lda, outer, outer),
                lda,
                block_ptr(A, lda, outer, next_inner),
                lda);
        });

    gemm_nn_update_m_tiled(
        handle,
        observer,
        "Inner GEMM",
        n - next_inner,
        next_size,
        history_size,
        block_ptr(A, lda, next_inner, outer),
        lda,
        block_ptr(A, lda, outer, next_inner),
        lda,
        block_ptr(A, lda, next_inner, next_inner),
        lda,
        inner_gemm_m_tile);

    return;
}

template <typename T, typename Observer>
void prepare_next_outer_panel(rocblas_handle handle,
                              Observer& observer,
                              T* A,
                              int lda,
                              int n,
                              int next_outer,
                              int outer_nb,
                              getrf_outer_trsm_mode outer_trsm_mode,
                              int trsm_block,
                              int outer_gemm_m_tile)
{
    using getrf_pivot_internal::block_ptr;

    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(outer_nb, n - next_outer);

    if (outer_trsm_mode == getrf_outer_trsm_direct) {
        observer.measure(
            "Outer TRSM direct",
            trsm_work(next_outer, next_size),
            [&]() {
                getrf_pivot_internal::trsm_left_lower_unit(
                    handle,
                    next_outer,
                    next_size,
                    A,
                    lda,
                    block_ptr(A, lda, 0, next_outer),
                    lda);
            });
    } else {
        trsm_left_lower_unit_iterative(
            handle,
            observer,
            next_outer,
            next_size,
            A,
            lda,
            block_ptr(A, lda, 0, next_outer),
            lda,
            trsm_block);
    }

    gemm_nn_update_m_tiled(
        handle,
        observer,
        "Outer GEMM",
        n - next_outer,
        next_size,
        next_outer,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, 0, next_outer),
        lda,
        block_ptr(A, lda, next_outer, next_outer),
        lda,
        outer_gemm_m_tile);

    return;
}

template <typename T, typename Observer>
bool double_blocked_getrf(rocblas_handle handle,
                          Observer& observer,
                          T* A,
                          int lda,
                          int n,
                          int outer_nb,
                          int inner_nb,
                          getrf_outer_trsm_mode outer_trsm_mode,
                          int trsm_block,
                          int inner_gemm_m_tile,
                          int outer_gemm_m_tile,
                          int* ipiv,
                          int* info,
                          getrf_batched_pivot_workspace& workspace)
{
    using getrf_pivot_internal::block_ptr;
    using getrf_pivot_internal::factor_panel;

    if (n <= 0) {
        return true;
    }
    if (outer_nb <= 0) {
        outer_nb = n;
    }
    if (inner_nb <= 0) {
        inner_nb = outer_nb;
    }
    inner_nb = std::min(inner_nb, outer_nb);

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));
    std::fill(workspace.host_pivots, workspace.host_pivots + n, 0);

    for (int outer = 0; outer < n; outer += outer_nb) {
        const int outer_size = std::min(outer_nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += inner_nb) {
            const int inner_size = std::min(inner_nb, outer_end - inner);
            const int next_inner = inner + inner_size;

            observer.measure(
                "Panel GETRF",
                panel_getrf_work(n - inner, inner_size),
                [&]() {
                    factor_panel(
                        handle,
                        n - inner,
                        inner_size,
                        block_ptr(A, lda, inner, inner),
                        lda,
                        ipiv + inner,
                        info);
                });

            bool pivot_valid = true;
            observer.measure(
                "Inner pivot batch",
                0.0,
                [&]() {
                    const int move_count = prepare_panel_permutation(
                        stream,
                        ipiv + inner,
                        inner,
                        inner_size,
                        n,
                        workspace);
                    if (move_count < 0) {
                        pivot_valid = false;
                        return;
                    }

                    // Match the reference schedule: apply this panel's pivots
                    // only to the already computed and still pending columns
                    // inside the current outer block.
                    if (inner > outer) {
                        apply_uploaded_permutation(
                            stream,
                            block_ptr(A, lda, inner, outer),
                            lda,
                            inner - outer,
                            move_count,
                            workspace);
                    }
                    if (next_inner < outer_end) {
                        apply_uploaded_permutation(
                            stream,
                            block_ptr(A, lda, inner, next_inner),
                            lda,
                            outer_end - next_inner,
                            move_count,
                            workspace);
                    }
                });
            if (!pivot_valid) {
                return false;
            }

            prepare_next_inner_panel(
                handle,
                observer,
                A,
                lda,
                n,
                outer,
                next_inner,
                outer_end,
                inner_nb,
                inner_gemm_m_tile);
        }

        bool pivot_valid = true;
        observer.measure(
            "Outer pivot batch",
            0.0,
            [&]() {
                // Current-block pivots complete the left history, while all
                // accumulated pivots prepare only the next outer panel.
                if (outer > 0) {
                    const int move_count = prepare_history_permutation(
                        stream,
                        outer,
                        outer_end,
                        outer,
                        n,
                        workspace);
                    if (move_count < 0) {
                        pivot_valid = false;
                        return;
                    }

                    apply_uploaded_permutation(
                        stream,
                        block_ptr(A, lda, outer, 0),
                        lda,
                        outer,
                        move_count,
                        workspace);
                }

                if (outer_end < n) {
                    const int move_count = prepare_history_permutation(
                        stream,
                        0,
                        outer_end,
                        0,
                        n,
                        workspace);
                    if (move_count < 0) {
                        pivot_valid = false;
                        return;
                    }

                    apply_uploaded_permutation(
                        stream,
                        block_ptr(A, lda, 0, outer_end),
                        lda,
                        std::min(outer_nb, n - outer_end),
                        move_count,
                        workspace);
                }
            });
        if (!pivot_valid) {
            return false;
        }

        prepare_next_outer_panel(
            handle,
            observer,
            A,
            lda,
            n,
            outer_end,
            outer_nb,
            outer_trsm_mode,
            trsm_block,
            outer_gemm_m_tile);
    }

    return true;
}

} // namespace getrf_batched_pivot_internal
