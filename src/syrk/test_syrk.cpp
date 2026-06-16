#include <cstdlib>

#include "syrk/benchmark/syrk_benchmark_common.h"

namespace {

ErrorMode parse_error_mode(int error_input)
{
    if (error_input == 1) {
        return ErrorMode::Forward;
    }
    if (error_input == 2) {
        return ErrorMode::Backward;
    }
    if (error_input == 3) {
        return ErrorMode::Both;
    }

    return ErrorMode::None;
}

} // namespace

int main(int argc, char** argv)
{
    syrk_benchmark::SyrkBenchmarkOptions options;
    int error_input = 0;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.k = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        error_input = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        options.test_recursive = std::atoi(argv[4]) != 0;
    }
    if (argc >= 6) {
        options.test_rocblas = std::atoi(argv[5]) != 0;
    }
    if (argc >= 7) {
        options.test_batched = std::atoi(argv[6]) != 0;
    }
    if (argc >= 8) {
        options.test_mixed = std::atoi(argv[7]) != 0;
    }

    for (int i = 8; i < argc; ++i) {
        options.nbs.push_back(std::atoi(argv[i]));
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    options.error_mode = parse_error_mode(error_input);
    syrk_benchmark::run_syrk_benchmarks(options);

    return 0;
}
