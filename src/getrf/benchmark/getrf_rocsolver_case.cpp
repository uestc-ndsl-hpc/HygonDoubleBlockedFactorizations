#include "getrf/benchmark/getrf_benchmark_common.h"

#include <type_traits>

#include <rocsolver/rocsolver.h>

#include "utils/common/error_check.h"

namespace getrf_benchmark {
namespace {

template <typename T>
void call_rocsolver_getrf(rocblas_handle handle,
                          int n,
                          T* A,
                          int lda,
                          bool pivoting,
                          int* ipiv,
                          int* info)
{
    if (pivoting) {
        if constexpr (std::is_same_v<T, float>) {
            ROCBLAS_CHECK(
                rocsolver_sgetrf(handle, n, n, A, lda, ipiv, info));
        } else {
            ROCBLAS_CHECK(
                rocsolver_dgetrf(handle, n, n, A, lda, ipiv, info));
        }
    } else {
        if constexpr (std::is_same_v<T, float>) {
            ROCBLAS_CHECK(
                rocsolver_sgetrf_npvt(handle, n, n, A, lda, info));
        } else {
            ROCBLAS_CHECK(
                rocsolver_dgetrf_npvt(handle, n, n, A, lda, info));
        }
    }

    return;
}

} // namespace

template <typename T>
void run_rocsolver_case(GetrfBenchmarkContext<T>& context)
{
    context.baseline_time = context.runner.execute(
        context.pivoting ? "rocSOLVER GETRF" : "rocSOLVER GETRF-NPVT",
        "N/A",
        context.reset,
        [&]() {
            call_rocsolver_getrf(
                context.handle,
                context.n,
                context.d_A,
                context.lda,
                context.pivoting,
                context.d_ipiv,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    context.results.back().speedup = 1.0;

    return;
}

template void run_rocsolver_case<float>(GetrfBenchmarkContext<float>& context);
template void run_rocsolver_case<double>(
    GetrfBenchmarkContext<double>& context);

} // namespace getrf_benchmark
