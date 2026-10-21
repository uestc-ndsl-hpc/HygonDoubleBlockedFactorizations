#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "potrf/benchmark/potrf_benchmark_common.h"
#include "utils/benchmark/seed.h"

namespace {

bool is_bool_flag(const char* arg)
{
    return std::strcmp(arg, "0") == 0 || std::strcmp(arg, "1") == 0;
}

bool parse_methods(const char* value,
                   potrf_benchmark::PotrfBenchmarkOptions& options)
{
    options.test_rocsolver = false;
    options.test_blocked = false;
    options.test_recursive = false;
    options.test_double_basic = false;
    options.test_double_dbtrsm = false;
    options.test_double_kbsyrk = false;
    options.test_double_mgemm = false;

    std::stringstream input(value);
    std::string method;
    bool found = false;
    while (std::getline(input, method, ',')) {
        if (method == "all") {
            options.test_rocsolver = true;
            options.test_blocked = true;
            options.test_recursive = true;
            options.test_double_basic = true;
            options.test_double_dbtrsm = true;
            options.test_double_kbsyrk = true;
            options.test_double_mgemm = true;
        } else if (method == "rocsolver") {
            options.test_rocsolver = true;
        } else if (method == "blocked") {
            options.test_blocked = true;
        } else if (method == "recursive") {
            options.test_recursive = true;
        } else if (method == "double" || method == "basic") {
            options.test_double_basic = true;
        } else if (method == "dbtrsm") {
            options.test_double_dbtrsm = true;
        } else if (method == "kbsyrk") {
            options.test_double_kbsyrk = true;
        } else if (method == "mgemm") {
            options.test_double_mgemm = true;
        } else {
            return false;
        }
        found = true;
    }

    options.test_double_blocked = options.test_double_basic
        || options.test_double_dbtrsm
        || options.test_double_kbsyrk
        || options.test_double_mgemm;

    return found;
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
    bool has_blocked_flag = false;
    bool has_double_blocked_flag = false;

    if (argc > nb_start && is_bool_flag(argv[nb_start])) {
        options.test_blocked = std::atoi(argv[nb_start]) != 0;
        has_blocked_flag = true;
        ++nb_start;
    }

    if (argc > nb_start && is_bool_flag(argv[nb_start])) {
        options.test_double_blocked = std::atoi(argv[nb_start]) != 0;
        has_double_blocked_flag = true;
        ++nb_start;
    }

    if (has_blocked_flag && !has_double_blocked_flag) {
        options.test_double_blocked = options.test_blocked;
    }

    enum class BlockParameter {
        Nb,
        DoubleNb,
        MicroBlock
    };

    BlockParameter block_parameter = BlockParameter::Nb;
    bool methods_explicit = false;
    for (int i = nb_start; i < argc; ++i) {
        if (std::strcmp(argv[i], "--skip-rec-tiled-trsm") == 0) {
            options.test_recursive_tiled_trsm = false;
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--cpu-fp64-residual") == 0) {
            options.cpu_fp64_residual = true;
            options.full_backward_error = true;
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--cpu-residual-tile") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --cpu-residual-tile.\n";
                return EXIT_FAILURE;
            }
            options.cpu_residual_tile = std::atoi(argv[++i]);
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || !parse_benchmark_seed(argv[++i], options.seed)) {
                std::cerr << "--seed requires an unsigned integer.\n";
                return EXIT_FAILURE;
            }
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--full-backward-error") == 0
            || std::strcmp(argv[i], "--full-residual") == 0) {
            options.full_backward_error = true;
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--full-backward-error-max-n") == 0
            || std::strcmp(argv[i], "--full-residual-max-n") == 0) {
            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value after --full-backward-error-max-n.\n";
                return EXIT_FAILURE;
            }
            options.full_backward_error_max_n = std::atoi(argv[++i]);
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--methods") == 0) {
            if (i + 1 >= argc || !parse_methods(argv[++i], options)) {
                std::cerr << "Invalid --methods value. Use a comma-separated "
                             "subset of rocsolver,blocked,recursive,double,"
                             "dbtrsm,kbsyrk,mgemm.\n";
                return EXIT_FAILURE;
            }
            methods_explicit = true;
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--syrk-k-tile") == 0
            || std::strcmp(argv[i], "--gemm-m-tile") == 0
            || std::strcmp(argv[i], "--gemm-m-threshold") == 0
            || std::strcmp(argv[i], "--gemm-k-threshold") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << argv[i] << ".\n";
                return EXIT_FAILURE;
            }

            const std::string option = argv[i];
            const int value = std::atoi(argv[++i]);
            if (option == "--syrk-k-tile") {
                options.syrk_k_tile = value;
            } else if (option == "--gemm-m-tile") {
                options.gemm_m_tile = value;
            } else if (option == "--gemm-m-threshold") {
                options.gemm_m_threshold = value;
            } else {
                options.gemm_k_threshold = value;
            }
            block_parameter = BlockParameter::Nb;
            continue;
        }
        if (std::strcmp(argv[i], "--double-nb") == 0) {
            block_parameter = BlockParameter::DoubleNb;
            continue;
        }
        if (std::strcmp(argv[i], "--b") == 0) {
            block_parameter = BlockParameter::MicroBlock;
            continue;
        }

        if (block_parameter == BlockParameter::DoubleNb) {
            options.double_nbs.push_back(std::atoi(argv[i]));
        } else if (block_parameter == BlockParameter::MicroBlock) {
            options.micro_blocks.push_back(std::atoi(argv[i]));
        } else {
            options.nbs.push_back(std::atoi(argv[i]));
        }
    }

    if (options.nbs.empty()) {
        options.nbs.push_back(256);
    }

    if (!methods_explicit && !options.test_double_blocked) {
        options.test_double_basic = false;
        options.test_double_dbtrsm = false;
        options.test_double_kbsyrk = false;
        options.test_double_mgemm = false;
    }
    if (options.syrk_k_tile < 0 || options.gemm_m_tile < 0
        || options.gemm_m_threshold < 0 || options.gemm_k_threshold < 0) {
        std::cerr << "POTRF tile and threshold values must be non-negative.\n";
        return EXIT_FAILURE;
    }

    options.error_mode = parse_benchmark_error_mode(error_input);
    if (options.cpu_residual_tile <= 0) {
        std::cerr << "--cpu-residual-tile must be positive.\n";
        return EXIT_FAILURE;
    }
    if (options.cpu_fp64_residual) {
        if (options.n <= 0 || options.precision_type != 0) {
            std::cerr << "--cpu-fp64-residual requires N > 0 and FP32 precision (0).\n";
            return EXIT_FAILURE;
        }
        if (options.error_mode == ErrorMode::Forward
            || options.error_mode == ErrorMode::Both) {
            std::cerr << "CPU residual mode requires error mode 0 or 2 (residual only); "
                         "forward/both would request a GPU FP64 reference POTRF.\n";
            return EXIT_FAILURE;
        }
        options.error_mode = ErrorMode::Backward;
    }
    if (options.full_backward_error) {
        if (options.full_backward_error_max_n <= 0
            || options.n > options.full_backward_error_max_n) {
            std::cerr << "Full normwise backward error requested for N="
                      << options.n
                      << ", but --full-backward-error-max-n="
                      << options.full_backward_error_max_n << ".\n";
            return EXIT_FAILURE;
        }
        if (options.error_mode == ErrorMode::None
            || options.error_mode == ErrorMode::Forward) {
            options.error_mode = options.error_mode == ErrorMode::Forward
                ? ErrorMode::Both
                : ErrorMode::Backward;
        }
    }
    potrf_benchmark::run_potrf_benchmarks(options);

    return 0;
}
