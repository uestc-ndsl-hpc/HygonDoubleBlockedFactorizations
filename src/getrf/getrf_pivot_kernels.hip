#include <algorithm>
#include <cstddef>

#include <hip/hip_runtime.h>

#include "getrf/getrf_pivot_kernels.h"
#include "utils/common/error_check.h"

namespace {

constexpr int kPermutationThreads = 256;
constexpr int kPermutationMaxBlocks = 65535;

template <typename T>
__global__ void apply_panel_pivots_kernel(T* A,
                                          int lda,
                                          int n,
                                          int panel,
                                          int panel_size,
                                          const int* pivots)
{
    const int external_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int external_columns = n - panel_size;
    if (external_index >= external_columns) {
        return;
    }

    const int column = external_index < panel
        ? external_index
        : external_index + panel_size;
    T* column_data = A + static_cast<size_t>(column) * lda;

    // A thread owns one column, so the LAPACK pivot sequence is race-free.
    for (int offset = 0; offset < panel_size; ++offset) {
        const int row = panel + offset;
        const int pivot_row = panel + pivots[offset] - 1;
        if (pivot_row != row) {
            const T value = column_data[row];
            column_data[row] = column_data[pivot_row];
            column_data[pivot_row] = value;
        }
    }

    return;
}

__global__ void globalize_panel_pivots_kernel(int* pivots,
                                               int panel,
                                               int panel_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < panel_size) {
        // rocSOLVER reports a 1-based row within A(panel:n, panel:*).
        pivots[index] += panel;
    }

    return;
}

template <typename T>
void launch_apply_panel_pivots(T* A,
                               int lda,
                               int n,
                               int panel,
                               int panel_size,
                               int* pivots,
                               hipStream_t stream)
{
    if (panel_size <= 0) {
        return;
    }

    constexpr int threads = 256;
    const int external_columns = n - panel_size;
    if (external_columns > 0) {
        const int blocks = (external_columns + threads - 1) / threads;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(apply_panel_pivots_kernel<T>),
            dim3(blocks),
            dim3(threads),
            0,
            stream,
            A,
            lda,
            n,
            panel,
            panel_size,
            pivots);
        HIP_CHECK(hipGetLastError());
    }

    const int pivot_blocks = (panel_size + threads - 1) / threads;
    hipLaunchKernelGGL(
        globalize_panel_pivots_kernel,
        dim3(pivot_blocks),
        dim3(threads),
        0,
        stream,
        pivots,
        panel,
        panel_size);
    HIP_CHECK(hipGetLastError());

    return;
}

template <typename T>
__global__ void gather_permuted_rows_kernel(const T* A,
                                             int lda,
                                             int columns,
                                             const int* sources,
                                             int move_count,
                                             T* temporary)
{
    const size_t total = static_cast<size_t>(columns) * move_count;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         index < total;
         index += stride) {
        const int move = static_cast<int>(index % move_count);
        const int column = static_cast<int>(index / move_count);
        temporary[index] = A[static_cast<size_t>(column) * lda + sources[move]];
    }

    return;
}

template <typename T>
__global__ void scatter_permuted_rows_kernel(T* A,
                                              int lda,
                                              int columns,
                                              const int* targets,
                                              int move_count,
                                              const T* temporary)
{
    const size_t total = static_cast<size_t>(columns) * move_count;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x
             + threadIdx.x;
         index < total;
         index += stride) {
        const int move = static_cast<int>(index % move_count);
        const int column = static_cast<int>(index / move_count);
        A[static_cast<size_t>(column) * lda + targets[move]] = temporary[index];
    }

    return;
}

template <typename T>
void launch_apply_permutation(T* A,
                              int lda,
                              int column_count,
                              const int* sources,
                              const int* targets,
                              int move_count,
                              T* temporary,
                              int column_tile,
                              hipStream_t stream)
{
    if (column_count <= 0 || move_count <= 0 || column_tile <= 0) {
        return;
    }

    for (int column = 0; column < column_count; column += column_tile) {
        const int columns = std::min(column_tile, column_count - column);
        const size_t total = static_cast<size_t>(columns) * move_count;
        const int blocks = static_cast<int>(std::min<size_t>(
            kPermutationMaxBlocks,
            (total + kPermutationThreads - 1) / kPermutationThreads));
        T* tile = A + static_cast<size_t>(column) * lda;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gather_permuted_rows_kernel<T>),
            dim3(blocks),
            dim3(kPermutationThreads),
            0,
            stream,
            tile,
            lda,
            columns,
            sources,
            move_count,
            temporary);
        HIP_CHECK(hipGetLastError());

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(scatter_permuted_rows_kernel<T>),
            dim3(blocks),
            dim3(kPermutationThreads),
            0,
            stream,
            tile,
            lda,
            columns,
            targets,
            move_count,
            temporary);
        HIP_CHECK(hipGetLastError());
    }

    return;
}

} // namespace

extern "C" {

void launch_apply_getrf_panel_pivots(float* A,
                                     int lda,
                                     int n,
                                     int panel,
                                     int panel_size,
                                     int* pivots,
                                     hipStream_t stream)
{
    launch_apply_panel_pivots(
        A, lda, n, panel, panel_size, pivots, stream);

    return;
}

void launch_apply_getrf_panel_pivots_double(double* A,
                                            int lda,
                                            int n,
                                            int panel,
                                            int panel_size,
                                            int* pivots,
                                            hipStream_t stream)
{
    launch_apply_panel_pivots(
        A, lda, n, panel, panel_size, pivots, stream);

    return;
}

void launch_apply_getrf_permutation(float* A,
                                    int lda,
                                    int column_count,
                                    const int* sources,
                                    const int* targets,
                                    int move_count,
                                    float* temporary,
                                    int column_tile,
                                    hipStream_t stream)
{
    launch_apply_permutation(
        A,
        lda,
        column_count,
        sources,
        targets,
        move_count,
        temporary,
        column_tile,
        stream);

    return;
}

void launch_apply_getrf_permutation_double(double* A,
                                           int lda,
                                           int column_count,
                                           const int* sources,
                                           const int* targets,
                                           int move_count,
                                           double* temporary,
                                           int column_tile,
                                           hipStream_t stream)
{
    launch_apply_permutation(
        A,
        lda,
        column_count,
        sources,
        targets,
        move_count,
        temporary,
        column_tile,
        stream);

    return;
}

} // extern "C"
