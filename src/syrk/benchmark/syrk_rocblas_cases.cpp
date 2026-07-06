#include "syrk/benchmark/syrk_benchmark_common.h"

#include "utils/common/error_check.h"

namespace syrk_benchmark {

void run_rocblas_case(SyrkBenchmarkContext& context)
{
    context.baseline_time = context.runner.execute(
        "rocBLAS (FP32)",
        "N/A",
        context.reset,
        [&]() {
            ROCBLAS_CHECK(rocblas_ssyrk(
                context.handle,
                rocblas_fill_lower,
                rocblas_operation_none,
                context.n,
                context.k,
                &context.alpha,
                context.d_A,
                context.lda,
                &context.beta,
                context.d_C,
                context.ldc));
        },
        context.verify,
        context.calc_gflops);

    context.results.back().speedup = 1.0;
}

} // namespace syrk_benchmark
