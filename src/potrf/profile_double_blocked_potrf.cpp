#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace {

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
const T* block_ptr(const T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

double potrf_work(int n)
{
    return static_cast<double>(n) * n * n / 3.0;
}

double trsm_work(int rows, int cols)
{
    return static_cast<double>(rows) * cols * cols;
}

double gemm_work(int rows, int cols, int k)
{
    return 2.0 * static_cast<double>(rows) * cols * k;
}

double syrk_work(int n, int k)
{
    return static_cast<double>(n) * n * k;
}

size_t tiled_syrk_operation_count(int n, int k, int tile, int k_tile)
{
    if (tile > 0) {
        const size_t block_count =
            (static_cast<size_t>(n) + static_cast<size_t>(tile) - 1) /
            static_cast<size_t>(tile);
        return block_count * (block_count + 1) / 2;
    }

    if (k_tile > 0) {
        return (static_cast<size_t>(k) + static_cast<size_t>(k_tile) - 1) /
               static_cast<size_t>(k_tile);
    }

    return 1;
}

size_t tile_operation_count(int extent, int tile)
{
    if (tile <= 0 || tile >= extent) {
        return 1;
    }

    return (static_cast<size_t>(extent) + static_cast<size_t>(tile) - 1) /
           static_cast<size_t>(tile);
}

struct OperationShapeEntry {
    std::string operation;
    int rows = 0;
    int cols = 0;
    int k = 0;
    double total_time_ms = 0.0;
    double total_work = 0.0;
    int call_count = 0;
};

struct PendingOperationSample {
    std::string operation;
    int rows = 0;
    int cols = 0;
    int k = 0;
    double work = 0.0;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
};

struct ProfilingEventPair {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
};

class OperationShapeProfiler {
public:
    OperationShapeProfiler() = default;
    OperationShapeProfiler(const OperationShapeProfiler&) = delete;
    OperationShapeProfiler& operator=(const OperationShapeProfiler&) = delete;

    ~OperationShapeProfiler()
    {
        for (const auto& sample : pending_) {
            hipEventDestroy(sample.start);
            hipEventDestroy(sample.stop);
        }
        for (const auto& events : available_events_) {
            hipEventDestroy(events.start);
            hipEventDestroy(events.stop);
        }
    }

    void reserve_events(size_t count)
    {
        while (available_events_.size() < count) {
            ProfilingEventPair events;
            HIP_CHECK(hipEventCreate(&events.start));
            HIP_CHECK(hipEventCreate(&events.stop));
            available_events_.push_back(events);
        }
    }

    void add_sample(const std::string& operation,
                    int rows,
                    int cols,
                    int k,
                    double elapsed_ms,
                    double work)
    {
        auto it = std::find_if(entries_.begin(), entries_.end(),
                               [&](const OperationShapeEntry& entry) {
                                   return entry.operation == operation &&
                                          entry.rows == rows &&
                                          entry.cols == cols &&
                                          entry.k == k;
                               });

        if (it == entries_.end()) {
            entries_.push_back({operation, rows, cols, k, elapsed_ms, work, 1});
            return;
        }

        it->total_time_ms += elapsed_ms;
        it->total_work += work;
        it->call_count += 1;
    }

    template <typename Func>
    void enqueue(hipStream_t stream,
                 const std::string& operation,
                 int rows,
                 int cols,
                 int k,
                 double work,
                 Func&& func)
    {
        if (rows <= 0 || cols <= 0 || k <= 0) {
            return;
        }

        PendingOperationSample sample;
        sample.operation = operation;
        sample.rows = rows;
        sample.cols = cols;
        sample.k = k;
        sample.work = work;

        if (available_events_.empty()) {
            reserve_events(1);
        }
        sample.start = available_events_.back().start;
        sample.stop = available_events_.back().stop;
        available_events_.pop_back();

        HIP_CHECK(hipEventRecord(sample.start, stream));
        func();
        HIP_CHECK(hipEventRecord(sample.stop, stream));
        pending_.push_back(sample);
    }

    void resolve_pending()
    {
        for (const auto& sample : pending_) {
            HIP_CHECK(hipEventSynchronize(sample.stop));

            float elapsed_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&elapsed_ms, sample.start, sample.stop));
            add_sample(sample.operation,
                       sample.rows,
                       sample.cols,
                       sample.k,
                       static_cast<double>(elapsed_ms),
                       sample.work);

            available_events_.push_back({sample.start, sample.stop});
        }

        pending_.clear();
    }

    void print(const std::string& title, int line_width = 126) const
    {
        using namespace std;

        cout << "\n" << string(line_width, '=') << "\n";
        cout << title << "\n";
        cout << string(line_width, '=') << "\n";
        cout << left << setw(22) << "Operation"
             << " | " << right << setw(8) << "M"
             << " | " << setw(8) << "N"
             << " | " << setw(8) << "K"
             << " | " << setw(10) << "Calls"
             << " | " << setw(12) << "Total(ms)"
             << " | " << setw(12) << "Avg(ms)"
             << " | " << setw(12) << "GFLOPS"
             << "\n";
        cout << string(line_width, '-') << "\n";

        for (const auto& entry : entries_) {
            const double avg_ms = entry.call_count > 0
                ? entry.total_time_ms / static_cast<double>(entry.call_count)
                : 0.0;
            const double gflops = entry.total_time_ms > 0.0
                ? (entry.total_work / (entry.total_time_ms / 1000.0)) / 1e9
                : 0.0;

            cout << left << setw(22) << entry.operation
                 << " | " << right << setw(8) << entry.rows
                 << " | " << setw(8) << entry.cols
                 << " | " << setw(8) << entry.k
                 << " | " << setw(10) << entry.call_count
                 << " | " << setw(12) << fixed << setprecision(3) << entry.total_time_ms
                 << " | " << setw(12) << fixed << setprecision(3) << avg_ms
                 << " | " << setw(12) << fixed << setprecision(2) << gflops
                 << "\n";
        }

        cout << string(line_width, '=') << "\n";
    }

private:
    std::vector<OperationShapeEntry> entries_;
    std::vector<PendingOperationSample> pending_;
    std::vector<ProfilingEventPair> available_events_;
};

template <typename Func>
void measure_gemm_shape(hipStream_t stream,
                        StageBreakdownProfiler& stage_profiler,
                        OperationShapeProfiler& shape_profiler,
                        const std::string& stage,
                        int rows,
                        int cols,
                        int k,
                        Func&& func)
{
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start, stream));
    func();
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    const double elapsed = static_cast<double>(elapsed_ms);
    const double work = gemm_work(rows, cols, k);
    stage_profiler.add_sample(stage, elapsed, work);
    shape_profiler.add_sample(stage, rows, cols, k, elapsed, work);
}

template <typename T>
void rocsolver_potrf(rocblas_handle handle, int n, T* A, int lda, int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(handle, rocblas_fill_lower, n, A, lda, info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, rocblas_fill_lower, n, A, lda, info));
    }

    return;
}

template <typename T>
void rocblas_trsm_right_lower_transpose(rocblas_handle handle,
                                        int rows,
                                        int cols,
                                        const T* A,
                                        int lda,
                                        T* B,
                                        int ldb)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_strsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            rows,
            cols,
            &one,
            A,
            lda,
            B,
            ldb));
    } else {
        ROCBLAS_CHECK(rocblas_dtrsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            rows,
            cols,
            &one,
            A,
            lda,
            B,
            ldb));
    }

    return;
}

template <typename T>
void rocblas_gemm_nt_update(rocblas_handle handle,
                            int rows,
                            int cols,
                            int k,
                            const T* A,
                            int lda,
                            const T* B,
                            int ldb,
                            T* C,
                            int ldc)
{
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            k,
            &neg_one,
            A,
            lda,
            B,
            ldb,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            k,
            &neg_one,
            A,
            lda,
            B,
            ldb,
            &one,
            C,
            ldc));
    }

    return;
}

template <typename T>
void rocblas_syrk_lower_update(rocblas_handle handle,
                               int n,
                               int k,
                               const T* A,
                               int lda,
                               T* C,
                               int ldc)
{
    if (n <= 0 || k <= 0) {
        return;
    }

    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_ssyrk(
            handle,
            rocblas_fill_lower,
            rocblas_operation_none,
            n,
            k,
            &neg_one,
            A,
            lda,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dsyrk(
            handle,
            rocblas_fill_lower,
            rocblas_operation_none,
            n,
            k,
            &neg_one,
            A,
            lda,
            &one,
            C,
            ldc));
    }

    return;
}

template <typename T>
void rocblas_tiled_syrk_lower_update_profiled(rocblas_handle handle,
                                              hipStream_t stream,
                                              OperationShapeProfiler& shape_profiler,
                                              int n,
                                              int k,
                                              const T* A,
                                              int lda,
                                              T* C,
                                              int ldc,
                                              int tile,
                                              int k_tile)
{
    if (n <= 0 || k <= 0) {
        return;
    }

    if (tile <= 0 && k_tile <= 0) {
        shape_profiler.enqueue(
            stream,
            "Outer SYRK API",
            n,
            n,
            k,
            syrk_work(n, k),
            [&]() {
                rocblas_syrk_lower_update(handle, n, k, A, lda, C, ldc);
            });
        return;
    }

    if (k_tile > 0) {
        const std::string operation =
            "Outer K-SYRK/" + std::to_string(k);

        for (int k_offset = 0; k_offset < k; k_offset += k_tile) {
            const int chunk = std::min(k_tile, k - k_offset);
            const T* A_chunk = block_ptr(A, lda, 0, k_offset);

            shape_profiler.enqueue(
                stream,
                operation,
                n,
                n,
                chunk,
                syrk_work(n, chunk),
                [&]() {
                    rocblas_syrk_lower_update(
                        handle, n, chunk, A_chunk, lda, C, ldc);
                });
        }
        return;
    }

    for (int col = 0; col < n; col += tile) {
        const int col_size = std::min(tile, n - col);
        const T* A_col = block_ptr(A, lda, col, 0);
        T* C_diag = block_ptr(C, ldc, col, col);

        shape_profiler.enqueue(
            stream,
            "Outer Diag SYRK",
            col_size,
            col_size,
            k,
            syrk_work(col_size, k),
            [&]() {
                rocblas_syrk_lower_update(
                    handle, col_size, k, A_col, lda, C_diag, ldc);
            });

        for (int row = col + col_size; row < n; row += tile) {
            const int row_size = std::min(tile, n - row);
            const T* A_row = block_ptr(A, lda, row, 0);
            T* C_block = block_ptr(C, ldc, row, col);

            shape_profiler.enqueue(
                stream,
                "Outer Offdiag GEMM",
                row_size,
                col_size,
                k,
                gemm_work(row_size, col_size, k),
                [&]() {
                    rocblas_gemm_nt_update(
                        handle,
                        row_size,
                        col_size,
                        k,
                        A_row,
                        lda,
                        A_col,
                        lda,
                        C_block,
                        ldc);
                });
        }
    }
}

template <typename T>
void update_next_inner_panel_profiled(rocblas_handle handle,
                                      hipStream_t stream,
                                      StageBreakdownProfiler& profiler,
                                      OperationShapeProfiler& shape_profiler,
                                      T* A,
                                      int lda,
                                      int n,
                                      int outer,
                                      int panel_start,
                                      int outer_end,
                                      int b,
                                      int gemm_k_tile,
                                      int gemm_m_tile,
                                      int gemm_m_threshold,
                                      int gemm_k_threshold)
{
    if (panel_start >= outer_end) {
        return;
    }

    const int panel_cols = std::min(b, outer_end - panel_start);
    const int rows = n - panel_start;
    const int k = panel_start - outer;

    if (gemm_m_tile > 0 && gemm_m_tile < rows &&
        rows > gemm_m_threshold && k >= gemm_k_threshold) {
        shape_profiler.reserve_events(
            tile_operation_count(rows, gemm_m_tile));

        profiler.measure(
            stream,
            "Inner GEMM",
            gemm_work(rows, panel_cols, k),
            [&]() {
                const std::string operation =
                    "Inner M-GEMM/" + std::to_string(rows);
                const T* B_panel =
                    block_ptr(A, lda, panel_start, outer);

                for (int row_offset = 0; row_offset < rows;
                     row_offset += gemm_m_tile) {
                    const int tile_rows =
                        std::min(gemm_m_tile, rows - row_offset);

                    shape_profiler.enqueue(
                        stream,
                        operation,
                        tile_rows,
                        panel_cols,
                        k,
                        gemm_work(tile_rows, panel_cols, k),
                        [&]() {
                            rocblas_gemm_nt_update(
                                handle,
                                tile_rows,
                                panel_cols,
                                k,
                                block_ptr(
                                    A, lda, panel_start + row_offset, outer),
                                lda,
                                B_panel,
                                lda,
                                block_ptr(
                                    A,
                                    lda,
                                    panel_start + row_offset,
                                    panel_start),
                                lda);
                        });
                }
            });
        shape_profiler.resolve_pending();
        return;
    }

    if (gemm_k_tile > 0 && gemm_k_tile < k) {
        shape_profiler.reserve_events(
            tile_operation_count(k, gemm_k_tile));

        profiler.measure(
            stream,
            "Inner GEMM",
            gemm_work(rows, panel_cols, k),
            [&]() {
                const std::string operation =
                    "Inner K-GEMM/" + std::to_string(k);

                for (int k_offset = 0; k_offset < k; k_offset += gemm_k_tile) {
                    const int chunk = std::min(gemm_k_tile, k - k_offset);
                    const T* A_chunk =
                        block_ptr(A, lda, panel_start, outer + k_offset);

                    shape_profiler.enqueue(
                        stream,
                        operation,
                        rows,
                        panel_cols,
                        chunk,
                        gemm_work(rows, panel_cols, chunk),
                        [&]() {
                            rocblas_gemm_nt_update(
                                handle,
                                rows,
                                panel_cols,
                                chunk,
                                A_chunk,
                                lda,
                                A_chunk,
                                lda,
                                block_ptr(A, lda, panel_start, panel_start),
                                lda);
                        });
                }
            });
        shape_profiler.resolve_pending();
        return;
    }

    measure_gemm_shape(
        stream,
        profiler,
        shape_profiler,
        "Inner GEMM",
        rows,
        panel_cols,
        k,
        [&]() {
            rocblas_gemm_nt_update(
                handle,
                rows,
                panel_cols,
                k,
                block_ptr(A, lda, panel_start, outer),
                lda,
                block_ptr(A, lda, panel_start, outer),
                lda,
                block_ptr(A, lda, panel_start, panel_start),
                lda);
        });

    return;
}

template <typename T>
void prepare_next_outer_panel_profiled(rocblas_handle handle,
                                       hipStream_t stream,
                                       StageBreakdownProfiler& profiler,
                                       OperationShapeProfiler& shape_profiler,
                                       T* A,
                                       int lda,
                                       int n,
                                       int next_outer,
                                       int nb,
                                       int syrk_tile,
                                       int syrk_k_tile)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(nb, n - next_outer);
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;

    shape_profiler.reserve_events(
        tiled_syrk_operation_count(
            next_size, next_outer, syrk_tile, syrk_k_tile));

    profiler.measure(
        stream,
        "Outer SYRK",
        syrk_work(next_size, next_outer),
        [&]() {
            rocblas_tiled_syrk_lower_update_profiled(
                handle,
                stream,
                shape_profiler,
                next_size,
                next_outer,
                block_ptr(A, lda, next_outer, 0),
                lda,
                block_ptr(A, lda, next_outer, next_outer),
                lda,
                syrk_tile,
                syrk_k_tile);
        });
    shape_profiler.resolve_pending();

    measure_gemm_shape(
        stream,
        profiler,
        shape_profiler,
        "Outer GEMM",
        below_rows,
        next_size,
        next_outer,
        [&]() {
            rocblas_gemm_nt_update(
                handle,
                below_rows,
                next_size,
                next_outer,
                block_ptr(A, lda, below_start, 0),
                lda,
                block_ptr(A, lda, next_outer, 0),
                lda,
                block_ptr(A, lda, below_start, next_outer),
                lda);
        });

    return;
}

template <typename T>
void double_blocked_potrf_profiled(rocblas_handle handle,
                                   hipStream_t stream,
                                   StageBreakdownProfiler& profiler,
                                   OperationShapeProfiler& shape_profiler,
                                   T* A,
                                   int lda,
                                   int n,
                                   int nb,
                                   int b,
                                   int syrk_tile,
                                   int syrk_k_tile,
                                   int inner_gemm_k_tile,
                                   int inner_gemm_m_tile,
                                   int inner_gemm_m_threshold,
                                   int inner_gemm_k_threshold,
                                   int* info)
{
    for (int outer = 0; outer < n; outer += nb) {
        const int outer_size = std::min(nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += b) {
            const int inner_size = std::min(b, outer_end - inner);
            const int panel_start = inner + inner_size;
            const int trailing_rows = n - panel_start;

            T* A_ii = block_ptr(A, lda, inner, inner);

            profiler.measure(
                stream,
                "POTRF",
                potrf_work(inner_size),
                [&]() {
                    rocsolver_potrf(handle, inner_size, A_ii, lda, info);
                });

            profiler.measure(
                stream,
                "TRSM",
                trsm_work(trailing_rows, inner_size),
                [&]() {
                    rocblas_trsm_right_lower_transpose(
                        handle,
                        trailing_rows,
                        inner_size,
                        A_ii,
                        lda,
                        block_ptr(A, lda, panel_start, inner),
                        lda);
                });

            update_next_inner_panel_profiled(
                handle,
                stream,
                profiler,
                shape_profiler,
                A,
                lda,
                n,
                outer,
                panel_start,
                outer_end,
                b,
                inner_gemm_k_tile,
                inner_gemm_m_tile,
                inner_gemm_m_threshold,
                inner_gemm_k_threshold);
        }

        prepare_next_outer_panel_profiled(
            handle,
            stream,
            profiler,
            shape_profiler,
            A,
            lda,
            n,
            outer_end,
            nb,
            syrk_tile,
            syrk_k_tile);
    }

    return;
}

int read_int_arg(int argc, char** argv, const std::string& name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name.c_str()) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }

    return fallback;
}

} // namespace

int main(int argc, char** argv)
{
    const int n = read_int_arg(argc, argv, "--n", 32768);
    const int nb = read_int_arg(argc, argv, "--nb", 8192);
    const int b = read_int_arg(argc, argv, "--b", 512);
    const int syrk_tile = read_int_arg(argc, argv, "--syrk-tile", 0);
    const int syrk_k_tile = read_int_arg(argc, argv, "--syrk-k-tile", 0);
    const int inner_gemm_k_tile =
        read_int_arg(argc, argv, "--inner-gemm-k-tile", 0);
    const int inner_gemm_m_tile =
        read_int_arg(argc, argv, "--inner-gemm-m-tile", 0);
    const int inner_gemm_m_threshold =
        read_int_arg(argc, argv, "--inner-gemm-m-threshold", 0);
    const int inner_gemm_k_threshold =
        read_int_arg(argc, argv, "--inner-gemm-k-threshold", 0);
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || nb <= 0 || b <= 0 || b > nb ||
        syrk_tile < 0 || syrk_tile > nb || syrk_k_tile < 0 ||
        inner_gemm_k_tile < 0 || inner_gemm_m_tile < 0 ||
        inner_gemm_m_threshold < 0 || inner_gemm_k_threshold < 0 ||
        (inner_gemm_k_tile > 0 && inner_gemm_m_tile > 0) ||
        (syrk_tile > 0 && syrk_k_tile > 0)) {
        std::cerr << "Usage: profile_double_blocked_potrf "
                  << "--n N --nb NB --b B [--syrk-tile T] "
                  << "[--syrk-k-tile KT] "
                  << "[--inner-gemm-k-tile GKT] "
                  << "[--inner-gemm-m-tile GMT] "
                  << "[--inner-gemm-m-threshold GMTM] "
                  << "[--inner-gemm-k-threshold GMTK] "
                  << "[--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t bytes = num_elements * sizeof(float);

    DeviceBuffer<float> d_A_orig(num_elements);
    DeviceBuffer<float> d_A(num_elements);
    DeviceBuffer<int> d_info(1);

    std::cout << "Generating SPD matrix: N=" << n << "\n";
    generate_spd_matrix<float>(handle, d_A_orig, n, n);

    std::cout << "Profiling DoubleBlock POTRF: N=" << n
              << ", nb=" << nb
              << ", b=" << b
              << ", syrk_tile=" << syrk_tile
              << ", syrk_k_tile=" << syrk_k_tile
              << ", inner_gemm_k_tile=" << inner_gemm_k_tile
              << ", inner_gemm_m_tile=" << inner_gemm_m_tile
              << ", inner_gemm_m_threshold=" << inner_gemm_m_threshold
              << ", inner_gemm_k_threshold=" << inner_gemm_k_threshold
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int i = 0; i < warmup; ++i) {
        StageBreakdownProfiler warmup_profiler;
        OperationShapeProfiler warmup_shape_profiler;
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        double_blocked_potrf_profiled(
            handle,
            stream,
            warmup_profiler,
            warmup_shape_profiler,
            d_A.get(),
            n,
            n,
            nb,
            b,
            syrk_tile,
            syrk_k_tile,
            inner_gemm_k_tile,
            inner_gemm_m_tile,
            inner_gemm_m_threshold,
            inner_gemm_k_threshold,
            d_info.get());
        warmup_shape_profiler.resolve_pending();
    }
    HIP_CHECK(hipDeviceSynchronize());

    StageBreakdownProfiler profiler("GFLOPS");
    OperationShapeProfiler shape_profiler;
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        double_blocked_potrf_profiled(
            handle,
            stream,
            profiler,
            shape_profiler,
            d_A.get(),
            n,
            n,
            nb,
            b,
            syrk_tile,
            syrk_k_tile,
            inner_gemm_k_tile,
            inner_gemm_m_tile,
            inner_gemm_m_threshold,
            inner_gemm_k_threshold,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());
    shape_profiler.resolve_pending();

    profiler.print(
        "DoubleBlock POTRF Stage Breakdown (N=" + std::to_string(n) +
        ", nb=" + std::to_string(nb) +
        ", b=" + std::to_string(b) +
        ", syrk_tile=" + std::to_string(syrk_tile) +
        ", syrk_k_tile=" + std::to_string(syrk_k_tile) +
        ", inner_gemm_k_tile=" + std::to_string(inner_gemm_k_tile) +
        ", inner_gemm_m_tile=" + std::to_string(inner_gemm_m_tile) +
        ", inner_gemm_m_threshold=" +
        std::to_string(inner_gemm_m_threshold) +
        ", inner_gemm_k_threshold=" +
        std::to_string(inner_gemm_k_threshold) + ")",
        110);

    shape_profiler.print(
        "DoubleBlock POTRF Operation Shape Breakdown (N=" + std::to_string(n) +
        ", nb=" + std::to_string(nb) +
        ", b=" + std::to_string(b) +
        ", syrk_tile=" + std::to_string(syrk_tile) +
        ", syrk_k_tile=" + std::to_string(syrk_k_tile) +
        ", inner_gemm_k_tile=" + std::to_string(inner_gemm_k_tile) +
        ", inner_gemm_m_tile=" + std::to_string(inner_gemm_m_tile) +
        ", inner_gemm_m_threshold=" +
        std::to_string(inner_gemm_m_threshold) +
        ", inner_gemm_k_threshold=" +
        std::to_string(inner_gemm_k_threshold) + ")");

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
