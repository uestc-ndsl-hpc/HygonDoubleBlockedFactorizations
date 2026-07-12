#include "potrf/inverse_panel_kernels.h"

#include <cstddef>

#include <hip/hip_runtime.h>

#include "utils/common/error_check.h"

namespace {

constexpr int kBlockSize = 16;

template <typename T>
__global__ void pack_lower_triangular_kernel(const T* source,
                                             int source_ld,
                                             T* destination,
                                             int destination_ld,
                                             int n)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= n || col >= n) {
        return;
    }

    const size_t destination_index =
        static_cast<size_t>(col) * destination_ld + row;
    if (row >= col) {
        const size_t source_index =
            static_cast<size_t>(col) * source_ld + row;
        destination[destination_index] = source[source_index];
    }

    return;
}

template <typename T>
__global__ void zero_strict_upper_triangle_kernel(T* matrix, int lda, int n)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n && row < col) {
        matrix[static_cast<size_t>(col) * lda + row] = T(0);
    }

    return;
}

template <typename T>
void launch_pack_lower_triangular(const T* source,
                                  int source_ld,
                                  T* destination,
                                  int destination_ld,
                                  int n,
                                  hipStream_t stream)
{
    if (n <= 0) {
        return;
    }

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(pack_lower_triangular_kernel<T>),
        grid,
        block,
        0,
        stream,
        source,
        source_ld,
        destination,
        destination_ld,
        n);
    HIP_CHECK(hipGetLastError());

    return;
}

template <typename T>
void launch_zero_strict_upper_triangle(T* matrix,
                                       int lda,
                                       int n,
                                       hipStream_t stream)
{
    if (n <= 1) {
        return;
    }

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(zero_strict_upper_triangle_kernel<T>),
        grid,
        block,
        0,
        stream,
        matrix,
        lda,
        n);
    HIP_CHECK(hipGetLastError());

    return;
}

__global__ void record_first_factorization_error_kernel(
    const int* local_info,
    int* global_info,
    int global_offset)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const int local = *local_info;
    if (local > 0) {
        atomicCAS(global_info, 0, global_offset + local);
    }

    return;
}

} // namespace

extern "C" {

void pack_lower_triangular_for_inverse(const float* source,
                                       int source_ld,
                                       float* destination,
                                       int destination_ld,
                                       int n,
                                       hipStream_t stream)
{
    launch_pack_lower_triangular(
        source, source_ld, destination, destination_ld, n, stream);

    return;
}

void pack_lower_triangular_for_inverse_double(const double* source,
                                              int source_ld,
                                              double* destination,
                                              int destination_ld,
                                              int n,
                                              hipStream_t stream)
{
    launch_pack_lower_triangular(
        source, source_ld, destination, destination_ld, n, stream);

    return;
}

void zero_strict_upper_triangle(float* matrix,
                                int lda,
                                int n,
                                hipStream_t stream)
{
    launch_zero_strict_upper_triangle(matrix, lda, n, stream);

    return;
}

void zero_strict_upper_triangle_double(double* matrix,
                                       int lda,
                                       int n,
                                       hipStream_t stream)
{
    launch_zero_strict_upper_triangle(matrix, lda, n, stream);

    return;
}

void record_first_factorization_error(const int* local_info,
                                      int* global_info,
                                      int global_offset,
                                      hipStream_t stream)
{
    hipLaunchKernelGGL(
        record_first_factorization_error_kernel,
        dim3(1),
        dim3(1),
        0,
        stream,
        local_info,
        global_info,
        global_offset);
    HIP_CHECK(hipGetLastError());

    return;
}

} // extern "C"
