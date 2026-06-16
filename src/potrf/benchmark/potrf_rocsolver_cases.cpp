#include "potrf/benchmark/potrf_benchmark_common.h"

#include <string>
#include <type_traits>

#include <rocsolver/rocsolver.h>

#include "utils/common/error_check.h"

namespace potrf_benchmark {
namespace {

template <typename T>
void call_rocsolver_potrf(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    T* A,
    int lda,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(handle, uplo, n, A, lda, info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, uplo, n, A, lda, info));
    }
}

} // namespace

template <typename T>
void run_rocsolver_case(PotrfBenchmarkContext<T>& context)
{
    std::string rocsolver_name =
        (context.precision_type == 2) ? "rocSOLVER (FP32)" : "rocSOLVER";

    context.baseline_time = context.runner.execute(
        rocsolver_name,
        "N/A",
        context.reset,
        [&]() {
            call_rocsolver_potrf<T>(
                context.handle,
                rocblas_fill_lower,
                context.n,
                context.d_A,
                context.lda,
                context.d_info);
        },
        context.verify,
        context.calc_gflops);

    context.results.back().speedup = 1.0;
}

template void run_rocsolver_case<float>(PotrfBenchmarkContext<float>& context);
template void run_rocsolver_case<double>(PotrfBenchmarkContext<double>& context);

} // namespace potrf_benchmark
