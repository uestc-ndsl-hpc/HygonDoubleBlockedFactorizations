#include "trsm/benchmark/trsm_benchmark_common.h"

#include <string>

#include "trsm/iterative_trsm.h"

namespace trsm_benchmark {

void run_streamed_cases(TrsmBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Streamed RL",
            std::to_string(nb),
            context.reset,
            [&]() {
                streamed_trsm_right_looking(
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
