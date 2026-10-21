#include "getrf/benchmark/getrf_benchmark_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>

#include "utils/benchmark/benchmark_printer.h"
#include "utils/benchmark/cpu_fp64_residual.h"
#include "utils/benchmark/residual_sampling.h"
#include "utils/common/error_check.h"
#include "utils/memory/device_buffer.h"

namespace getrf_benchmark {
namespace {

double calculate_gflops(int n, double time_ms)
{
    if (time_ms <= 0.0) {
        return 0.0;
    }

    const double dn = static_cast<double>(n);
    const double flops = (2.0 / 3.0) * dn * dn * dn;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

template <typename T>
void generate_dense_matrix(std::vector<T>& A, int n, bool pivoting, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (T& value : A) {
        value = static_cast<T>(dist(gen));
    }
    if (!pivoting) {
        for (int i = 0; i < n; ++i) {
            A[static_cast<size_t>(i) * n + i] += static_cast<T>(n);
        }
    }

    return;
}

bool build_row_permutation(const int* pivots,
                           int n,
                           bool pivoting,
                           std::vector<int>& permutation)
{
    permutation.resize(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    if (pivoting) {
        for (int row = 0; row < n; ++row) {
            const int pivot_row = pivots[row] - 1;
            if (pivot_row < row || pivot_row >= n) {
                return false;
            }
            std::swap(permutation[row], permutation[pivot_row]);
        }
    }

    return true;
}

template <typename T>
void gemm_nn(rocblas_handle handle,
             int n,
             const T* left,
             const T* right,
             T* product)
{
    const T one = static_cast<T>(1);
    const T zero = static_cast<T>(0);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            n,
            n,
            n,
            &one,
            left,
            n,
            right,
            n,
            &zero,
            product,
            n));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            n,
            n,
            n,
            &one,
            left,
            n,
            right,
            n,
            &zero,
            product,
            n));
    }
}

template <typename T>
double verify_normwise_backward_error_full(rocblas_handle handle,
                                           const T* packed_lu,
                                           const T* original,
                                           const int* pivots,
                                           int n,
                                           int lda,
                                           bool pivoting,
                                           T* device_l,
                                           T* device_u,
                                           T* device_product)
{
    // For fixed P, L, and U, P*(A + DeltaA) = L*U uniquely determines
    // DeltaA = P^T*L*U - A. Frobenius-norm invariance under P gives the
    // relative normwise backward error ||P*A-L*U||_F / ||A||_F.
    std::vector<int> permutation;
    if (!build_row_permutation(
            pivots, n, pivoting, permutation)) {
        return std::numeric_limits<double>::infinity();
    }

    const size_t element_count = static_cast<size_t>(n) * n;
    const size_t bytes = element_count * sizeof(T);
    std::vector<T> lower(element_count, static_cast<T>(0));
    std::vector<T> upper(element_count, static_cast<T>(0));

    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            const size_t source_index
                = static_cast<size_t>(column) * lda + row;
            const size_t dense_index
                = static_cast<size_t>(column) * n + row;
            if (row > column) {
                lower[dense_index] = packed_lu[source_index];
            } else if (row == column) {
                lower[dense_index] = static_cast<T>(1);
                upper[dense_index] = packed_lu[source_index];
            } else {
                upper[dense_index] = packed_lu[source_index];
            }
        }
    }

    HIP_CHECK(hipMemcpy(
        device_l, lower.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        device_u, upper.data(), bytes, hipMemcpyHostToDevice));
    gemm_nn<T>(handle, n, device_l, device_u, device_product);

    std::vector<T> product(element_count);
    HIP_CHECK(hipMemcpy(
        product.data(), device_product, bytes, hipMemcpyDeviceToHost));

    double residual_norm_sq = 0.0;
    double original_norm_sq = 0.0;
    for (int column = 0; column < n; ++column) {
        for (int row = 0; row < n; ++row) {
            const size_t dense_index
                = static_cast<size_t>(column) * n + row;
            const double pa_value = static_cast<double>(
                original[static_cast<size_t>(column) * lda
                    + permutation[row]]);
            const double residual
                = static_cast<double>(product[dense_index]) - pa_value;
            residual_norm_sq += residual * residual;
            original_norm_sq += pa_value * pa_value;
        }
    }

    return std::sqrt(residual_norm_sq)
        / (std::sqrt(original_norm_sq) + 1e-30);
}

template <typename T>
double verify_sampled_relative_residual(const T* packed_lu,
                                        const T* original,
                                        const int* pivots,
                                        int n,
                                        int lda,
                                        bool pivoting,
                                        int sample_count = 4096)
{
    const std::uint64_t total
        = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n);
    sample_count = static_cast<int>(
        std::min<std::uint64_t>(sample_count, total));
    benchmark_residual::DeterministicIndexSampler sampler(total);

    std::vector<int> permutation;
    if (!build_row_permutation(
            pivots, n, pivoting, permutation)) {
        return std::numeric_limits<double>::infinity();
    }

    double residual_norm_sq = 0.0;
    double original_norm_sq = 0.0;

    for (int sample = 0; sample < sample_count; ++sample) {
        const std::uint64_t index = sampler.next();
        const int row = static_cast<int>(index % n);
        const int col = static_cast<int>(index / n);

        double reconstructed = 0.0;
        const int k_end = std::min(row, col);
        for (int k = 0; k <= k_end; ++k) {
            const double l_value = (row == k)
                ? 1.0
                : static_cast<double>(
                    packed_lu[static_cast<size_t>(k) * lda + row]);
            const double u_value = static_cast<double>(
                packed_lu[static_cast<size_t>(col) * lda + k]);
            reconstructed += l_value * u_value;
        }

        const int original_row = permutation[row];
        const double a_value = static_cast<double>(
            original[static_cast<size_t>(col) * lda + original_row]);
        const double residual = reconstructed - a_value;
        residual_norm_sq += residual * residual;
        original_norm_sq += a_value * a_value;
    }

    return std::sqrt(residual_norm_sq)
        / (std::sqrt(original_norm_sq) + 1e-30);
}

template <typename T>
void run_all_benchmarks(const GetrfBenchmarkOptions& options)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const int n = options.n;
    const size_t element_count = static_cast<size_t>(n) * n;
    const size_t bytes = element_count * sizeof(T);

    std::vector<T> h_A_original(element_count);
    generate_dense_matrix(h_A_original, n, options.pivoting, options.seed);

    DeviceBuffer<T> d_A_original(element_count);
    DeviceBuffer<T> d_A(element_count);
    DeviceBuffer<int> d_ipiv(n);
    DeviceBuffer<int> d_info(1);
    const bool device_residual = options.full_backward_error
        && !options.cpu_fp64_residual;
    DeviceBuffer<T> d_residual_lhs(
        device_residual ? element_count : 0);
    DeviceBuffer<T> d_residual_rhs(
        device_residual ? element_count : 0);
    DeviceBuffer<T> d_residual_product(
        device_residual ? element_count : 0);

    const int pivot_column_tile = std::min(
        n, std::max(1, options.pivot_column_tile));
    const size_t pivot_capacity = options.pivoting
        ? static_cast<size_t>(n)
        : 0;
    DeviceBuffer<int> d_pivot_sources(pivot_capacity);
    DeviceBuffer<int> d_pivot_targets(pivot_capacity);
    DeviceBuffer<T> d_pivot_swap(
        pivot_capacity * static_cast<size_t>(pivot_column_tile));
    std::vector<int> h_pivots(pivot_capacity);
    std::vector<int> h_permutation(pivot_capacity);
    std::vector<int> h_sources(pivot_capacity);
    std::vector<int> h_targets(pivot_capacity);

    getrf_batched_pivot_workspace batched_pivot_workspace{
        d_pivot_sources.get(),
        d_pivot_targets.get(),
        d_pivot_swap.get(),
        h_pivots.data(),
        h_permutation.data(),
        h_sources.data(),
        h_targets.data(),
        options.pivoting ? n : 0,
        pivot_column_tile
    };

    HIP_CHECK(hipMemcpy(
        d_A_original,
        h_A_original.data(),
        bytes,
        hipMemcpyHostToDevice));

    std::vector<BenchmarkResult> results;
    double baseline_time = -1.0;
    const ResidualMetric residual_metric = options.error_mode == ErrorMode::None
        ? ResidualMetric::None
        : (options.cpu_fp64_residual
              ? ResidualMetric::CpuFp64RelativeResidual
              : options.full_backward_error
              ? ResidualMetric::NormwiseBackwardError
              : ResidualMetric::SampledRelativeResidual);
    BenchmarkRunner runner(stream, baseline_time, results, residual_metric);

    auto reset = [&]() {
        HIP_CHECK(hipMemcpy(
            d_A, d_A_original, bytes, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemset(d_ipiv, 0, static_cast<size_t>(n) * sizeof(int)));
        HIP_CHECK(hipMemset(d_info, 0, sizeof(int)));
    };

    auto verify = [&](double& forward_error, double& residual_value) {
        (void)forward_error;
        if (options.error_mode == ErrorMode::None) {
            return;
        }

        int info = 0;
        HIP_CHECK(hipMemcpy(
            &info, d_info, sizeof(int), hipMemcpyDeviceToHost));
        if (options.cpu_fp64_residual) {
            std::cout << ">>> CPU FP64 validation: info=" << info << "\n";
        }
        if (info != 0) {
            residual_value = std::numeric_limits<double>::infinity();
            return;
        }

        std::vector<T> h_lu(element_count);
        HIP_CHECK(hipMemcpy(
            h_lu.data(), d_A, bytes, hipMemcpyDeviceToHost));

        std::vector<int> h_ipiv;
        if (options.pivoting) {
            h_ipiv.resize(n);
            HIP_CHECK(hipMemcpy(
                h_ipiv.data(),
                d_ipiv,
                static_cast<size_t>(n) * sizeof(int),
                hipMemcpyDeviceToHost));
        }

        if (options.cpu_fp64_residual) {
            residual_value = benchmark_residual::getrf_cpu_fp64_full(
                h_lu.data(), h_A_original.data(),
                options.pivoting ? h_ipiv.data() : nullptr,
                n, n, options.pivoting, options.cpu_residual_tile);
        } else if (options.full_backward_error) {
            residual_value = verify_normwise_backward_error_full(
                handle,
                h_lu.data(),
                h_A_original.data(),
                options.pivoting ? h_ipiv.data() : nullptr,
                n,
                n,
                options.pivoting,
                d_residual_lhs.get(),
                d_residual_rhs.get(),
                d_residual_product.get());
        } else {
            residual_value = verify_sampled_relative_residual(
                h_lu.data(),
                h_A_original.data(),
                options.pivoting ? h_ipiv.data() : nullptr,
                n,
                n,
                options.pivoting);
        }
    };

    auto calc_gflops = [n](double time_ms) {
        return calculate_gflops(n, time_ms);
    };

    GetrfBenchmarkContext<T> context{
        handle,
        d_A,
        d_ipiv,
        d_info,
        n,
        n,
        options.pivoting,
        options.test_double_basic,
        options.test_batched_pivot,
        options.test_mgemm,
        options.outer_trsm_mode,
        options.trsm_block,
        pivot_column_tile,
        options.inner_gemm_m_tile,
        options.pivoting ? &batched_pivot_workspace : nullptr,
        options.nbs,
        options.double_nbs,
        options.micro_blocks,
        runner,
        results,
        baseline_time,
        reset,
        verify,
        calc_gflops
    };

    const std::string mode = options.pivoting ? "GETRF" : "GETRF-NPVT";
    std::cout << ">>> Running " << mode << " Benchmarks...\n";
    if (options.error_mode != ErrorMode::None) {
        if (options.cpu_fp64_residual) {
            std::cout << ">>> Full relative factorization residual "
                         "||P*A-L*U||_F/||A||_F: CPU FP64 product, subtraction "
                         "and norms; " << benchmark_residual::cpu_fp64_backend_name()
                      << "; tile=" << options.cpu_residual_tile
                      << "; seed=" << options.seed
                      << "; validation excluded from GPU timing\n";
        } else if (options.full_backward_error) {
            std::cout
                << ">>> Normwise backward error "
                   "eta_F=||P*A-L*U||_F/||A||_F: full matrix, "
                   "factorization-precision product, FP64 norm accumulation\n";
        } else {
            std::cout
                << ">>> Sampled relative residual "
                   "sqrt(sum_S|P*A-L*U|^2)/sqrt(sum_S|A|^2): "
                   "4096 deterministic entries\n";
        }
    }
    if (options.test_rocsolver) {
        run_rocsolver_case(context);
    }
    if (options.test_blocked) {
        run_blocked_cases(context);
    }
    if (options.test_double_basic
        || (options.pivoting
            && (options.test_batched_pivot || options.test_mgemm))) {
        run_double_blocked_cases(context);
    }

    print_benchmark_results(
        mode + " Benchmark Summary (N = " + std::to_string(n) + ")",
        results,
        options.error_mode,
        120);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

} // namespace

void run_getrf_benchmarks(const GetrfBenchmarkOptions& options)
{
    if (options.cpu_fp64_residual
        && (options.precision_type != 0 || options.n <= 0
            || options.cpu_residual_tile <= 0 || !options.full_backward_error
            || options.error_mode != ErrorMode::Backward
            || options.full_backward_error_max_n < options.n)) {
        throw std::invalid_argument(
            "CPU FP64 residual requires FP32 factors, residual-only mode, "
            "a positive tile, and an explicit full-matrix size limit");
    }
    if (options.precision_type == 1) {
        std::cout << "Running "
                  << (options.pivoting ? "GETRF" : "GETRF-NPVT")
                  << " benchmark in DOUBLE precision\n";
        run_all_benchmarks<double>(options);
    } else {
        std::cout << "Running "
                  << (options.pivoting ? "GETRF" : "GETRF-NPVT")
                  << " benchmark in FLOAT precision\n";
        run_all_benchmarks<float>(options);
    }

    return;
}

} // namespace getrf_benchmark
