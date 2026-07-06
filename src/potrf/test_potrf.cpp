#include <cstdlib>
#include <cstring>
#include <vector>

#include "potrf/benchmark/potrf_benchmark_common.h"

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

    int nb_start = 6;
    if (argc >= 7 &&
        (std::strcmp(argv[6], "0") == 0 || std::strcmp(argv[6], "1") == 0)) {
        options.test_blocked = std::atoi(argv[6]) != 0;
        nb_start = 7;
    }

    for (int i = nb_start; i < argc; ++i) {
        options.nbs.push_back(std::atoi(argv[i]));
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    options.error_mode = parse_error_mode(error_input);
    potrf_benchmark::run_potrf_benchmarks(options);

    return 0;
}
