#include <cstdlib>
#include <cstring>

#include "trsm/benchmark/trsm_benchmark_common.h"

int main(int argc, char** argv)
{
    trsm_benchmark::TrsmBenchmarkOptions options;
    int error_input = 0;

    if (argc >= 2) {
        options.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        options.m = std::atoi(argv[2]);
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
        options.test_iterative = std::atoi(argv[6]) != 0;
    }

    if (argc >= 8) {
        options.test_half = std::atoi(argv[7]) != 0;
        int nb_start = 8;
        if (argc >= 9 && (std::strcmp(argv[8], "0") == 0 || std::strcmp(argv[8], "1") == 0)) {
            options.test_streamed = std::atoi(argv[8]) != 0;
            nb_start = 9;
        }
        for (int i = nb_start; i < argc; ++i) {
            options.nbs.push_back(std::atoi(argv[i]));
        }
    } else {
        for (int i = 7; i < argc; ++i) {
            options.nbs.push_back(std::atoi(argv[i]));
        }
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    options.error_mode = parse_benchmark_error_mode(error_input);
    trsm_benchmark::run_trsm_benchmarks(options);

    return 0;
}
