#include "syrk/benchmark/syrk_benchmark_common.h"

#include <string>

#include "syrk/batched_syrk.h"

namespace syrk_benchmark {

void run_paper_cases(SyrkBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "PaperSched (FP32)",
            std::to_string(nb),
            context.reset,
            [&]() {
                batched_syrk_paper_schedule(
                    context.handle,
                    context.pool,
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
