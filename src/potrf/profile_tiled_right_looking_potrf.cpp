#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "gemm/update_gemm.h"
#include "potrf/blocked_potrf.h"
#include "potrf/tiled_right_looking_potrf.h"
#include "syrk/kblocked_syrk.h"
#include "trsm/recursive_trsm.h"
#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace {

float* block_ptr(float* A, int lda, int row, int col)
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

double syrk_work(int n, int k)
{
    return static_cast<double>(n) * n * k;
}

double gemm_work(int rows, int cols, int k)
{
    return 2.0 * static_cast<double>(rows) * cols * k;
}

void tiled_right_looking_potrf_profiled(
    rocblas_handle handle,
    hipStream_t stream,
    StageBreakdownProfiler& profiler,
    float* A,
    int lda,
    int n,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info)
{
    for (int outer = 0; outer < n; outer += outer_block_size) {
        const int outer_size = std::min(outer_block_size, n - outer);
        const int trailing_start = outer + outer_size;
        const int trailing_size = n - trailing_start;
        float* diagonal = block_ptr(A, lda, outer, outer);

        profiler.measure(
            stream,
            "Macro POTRF",
            potrf_work(outer_size),
            [&]() {
                blocked_right_looking_potrf(
                    handle,
                    diagonal,
                    lda,
                    outer_size,
                    std::min(potrf_leaf_size, outer_size),
                    info);
            });

        if (trailing_size <= 0) {
            continue;
        }

        float* panel = block_ptr(A, lda, trailing_start, outer);

        profiler.measure(
            stream,
            "Recursive TRSM",
            trsm_work(trailing_size, outer_size),
            [&]() {
                recursive_trsm_tiled_update(
                    handle,
                    diagonal,
                    lda,
                    panel,
                    lda,
                    trailing_size,
                    outer_size,
                    std::min(trsm_leaf_size, outer_size));
            });

        float* trailing = block_ptr(A, lda, trailing_start, trailing_start);

        for (int k_offset = 0; k_offset < outer_size;
             k_offset += update_k_tile) {
            const int chunk =
                std::min(update_k_tile, outer_size - k_offset);

            for (int col = 0; col < trailing_size;
                 col += update_tile_size) {
                const int cols =
                    std::min(update_tile_size, trailing_size - col);
                const float* panel_col =
                    block_ptr(panel, lda, col, k_offset);

                profiler.measure(
                    stream,
                    "Diagonal SYRK",
                    syrk_work(cols, chunk),
                    [&]() {
                        syrk_lower_update(
                            handle,
                            cols,
                            chunk,
                            panel_col,
                            lda,
                            block_ptr(trailing, lda, col, col),
                            lda);
                    });

                for (int row = col + cols; row < trailing_size;
                     row += update_tile_size) {
                    const int rows =
                        std::min(update_tile_size, trailing_size - row);

                    profiler.measure(
                        stream,
                        "Offdiag GEMM",
                        gemm_work(rows, cols, chunk),
                        [&]() {
                            gemm_nt_update(
                                handle,
                                rows,
                                cols,
                                chunk,
                                block_ptr(panel, lda, row, k_offset),
                                lda,
                                panel_col,
                                lda,
                                block_ptr(trailing, lda, row, col),
                                lda);
                        });
                }
            }
        }
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
    const int n = read_int_arg(argc, argv, "--n", 40960);
    const int outer_block_size = read_int_arg(argc, argv, "--kb", 8192);
    const int update_tile_size = read_int_arg(argc, argv, "--tile", 4096);
    const int potrf_leaf_size = read_int_arg(argc, argv, "--leaf", 1024);
    const int trsm_leaf_size =
        read_int_arg(argc, argv, "--trsm-leaf", 1024);
    const int update_k_tile =
        read_int_arg(argc, argv, "--k-tile", 8192);
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || outer_block_size <= 0 || update_tile_size <= 0 ||
        potrf_leaf_size <= 0 || trsm_leaf_size <= 0 ||
        update_k_tile <= 0 || warmup < 0 || iters <= 0) {
        std::cerr << "Usage: profile_tiled_right_looking_potrf "
                  << "--n N --kb KB --tile T --leaf IB "
                  << "--trsm-leaf TB --k-tile KT "
                  << "[--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_pointer_mode(
        handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t bytes = num_elements * sizeof(float);

    DeviceBuffer<float> d_A_orig(num_elements);
    DeviceBuffer<float> d_A(num_elements);
    DeviceBuffer<int> d_info(1);

    std::cout << "Generating SPD matrix: N=" << n << "\n";
    generate_spd_matrix<float>(handle, d_A_orig, n, n);

    std::cout << "Profiling Tiled-RL DoubleBlock POTRF: N=" << n
              << ", KB=" << outer_block_size
              << ", tile=" << update_tile_size
              << ", leaf=" << potrf_leaf_size
              << ", trsm_leaf=" << trsm_leaf_size
              << ", k_tile=" << update_k_tile
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int i = 0; i < warmup; ++i) {
        HIP_CHECK(hipMemcpy(
            d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        tiled_right_looking_double_blocked_potrf(
            handle,
            d_A.get(),
            n,
            n,
            outer_block_size,
            update_tile_size,
            potrf_leaf_size,
            trsm_leaf_size,
            update_k_tile,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    StageBreakdownProfiler profiler("GFLOPS");
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipMemcpy(
            d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
        tiled_right_looking_potrf_profiled(
            handle,
            stream,
            profiler,
            d_A.get(),
            n,
            n,
            outer_block_size,
            update_tile_size,
            potrf_leaf_size,
            trsm_leaf_size,
            update_k_tile,
            d_info.get());
    }
    HIP_CHECK(hipDeviceSynchronize());

    profiler.print(
        "Tiled-RL DoubleBlock POTRF Stage Breakdown (N=" +
        std::to_string(n) + ", KB=" +
        std::to_string(outer_block_size) + ", tile=" +
        std::to_string(update_tile_size) + ", leaf=" +
        std::to_string(potrf_leaf_size) + ")",
        112);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return EXIT_SUCCESS;
}
