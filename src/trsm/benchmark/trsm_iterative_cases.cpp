#include "trsm/benchmark/trsm_benchmark_common.h"

#include <string>

#include "trsm/iterative_trsm.h"

namespace trsm_benchmark {

void run_iterative_cases(TrsmBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Iterative RL",
            std::to_string(nb),
            context.reset,
            [&]() {
                iterative_trsm_right_looking(
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
}

} // namespace trsm_benchmark
