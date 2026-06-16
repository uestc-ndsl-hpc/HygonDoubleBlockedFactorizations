#include "syrk/benchmark/syrk_benchmark_common.h"

#include <string>

#include "syrk/recursive_syrk.h"

namespace syrk_benchmark {

void run_recursive_cases(SyrkBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Recursive (FP32)",
            std::to_string(nb),
            context.reset,
            [&]() {
                recursive_syrk_parallel(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.d_C,
                    context.ldc,
                    context.n,
                    context.k,
                    nb);
            },
            context.verify,
            context.calc_gflops);
    }

    return;
}

} // namespace syrk_benchmark
