#include "syrk/benchmark/syrk_benchmark_common.h"

#include <string>

#include "syrk/batched_syrk.h"

namespace syrk_benchmark {

void run_mixed_cases(SyrkBenchmarkContext& context)
{
    for (int nb : context.nbs) {
        context.runner.execute(
            "Batched (Mixed)",
            std::to_string(nb),
            context.reset,
            [&]() {
                batched_syrk_bottom_up_mixed(
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
}

} // namespace syrk_benchmark
