#include "trsm/benchmark/trsm_benchmark_common.h"

#include <string>

#include "trsm/recursive_trsm.h"

namespace trsm_benchmark {

void run_mixed_cases(TrsmBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Rec (FP16)",
            std::to_string(nb),
            context.reset,
            [&]() {
                recursive_trsm_mixed(
                    context.handle,
                    context.pool,
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
