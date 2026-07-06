#include <cstdlib>
#include <vector>

#include "potrf/benchmark/potrf_benchmark_common.h"

int main(int argc, char** argv)
{
    potrf_benchmark::PotrfBenchmarkOptions options;
    int error_input = 0;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.test_rocsolver = std::atoi(argv[2]) != 0;
    }
    if (argc >= 4) {
        options.test_recursive = std::atoi(argv[3]) != 0;
    }
    if (argc >= 5) {
        error_input = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        options.precision_type = std::atoi(argv[5]);
    }

    for (int i = 6; i < argc; ++i) {
        options.nbs.push_back(std::atoi(argv[i]));
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    options.error_mode = parse_benchmark_error_mode(error_input);
    potrf_benchmark::run_potrf_benchmarks(options);

    return 0;
}
