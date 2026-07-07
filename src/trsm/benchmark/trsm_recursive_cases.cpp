#include "trsm/benchmark/trsm_benchmark_common.h"

#include <string>

#include "trsm/recursive_trsm.h"

namespace trsm_benchmark {

void run_recursive_cases(TrsmBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Rec (FP32)",
            std::to_string(nb),
            context.reset,
            [&]() {
                recursive_trsm(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.d_B,
                    context.ldb,
                    context.n,
                    context.m,
                    nb);
            },
            context.verify,
            context.calc_gflops);

        context.runner.execute(
            "RecTiled8K (FP32)",
            std::to_string(nb),
            context.reset,
            [&]() {
                recursive_trsm_tiled_update(
                    context.handle,
                    context.d_A,
                    context.lda,
                    context.d_B,
                    context.ldb,
                    context.n,
                    context.m,
                    nb);
            },
            context.verify,
            context.calc_gflops);
    }

    return;
}

} // namespace trsm_benchmark
