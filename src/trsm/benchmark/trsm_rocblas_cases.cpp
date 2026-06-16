#include "trsm/benchmark/trsm_benchmark_common.h"

#include "utils/common/error_check.h"

namespace trsm_benchmark {

void run_rocblas_case(TrsmBenchmarkContext& context)
{
    context.baseline_time = context.runner.execute(
        "rocBLAS (FP32)",
        "N/A",
        context.reset,
        [&]() {
            ROCBLAS_CHECK(rocblas_strsm(
                context.handle,
                rocblas_side_right,
                rocblas_fill_lower,
                rocblas_operation_transpose,
                rocblas_diagonal_non_unit,
                context.n,
                context.m,
                &context.alpha,
                context.d_A,
                context.lda,
                context.d_B,
                context.ldb));
        },
        context.verify,
        context.calc_gflops);

    context.results.back().speedup = 1.0;
}

} // namespace trsm_benchmark
