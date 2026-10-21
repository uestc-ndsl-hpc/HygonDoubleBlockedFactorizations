#include "utils/benchmark/cpu_fp64_residual.h"
#include "utils/benchmark/residual_metric.h"
#include "utils/benchmark/seed.h"

#include <cstring>
#include <iostream>
#include <string>

namespace {
int checks = 0;

void require(bool condition, const std::string& label)
{
    ++checks;
    if (!condition) throw std::runtime_error(label);
}

void close_to(double actual, long double expected, const std::string& label)
{
    const long double tolerance = std::max(1.0e-18L, std::abs(expected) * 1.0e-7L);
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, label);
}

// Independent, unblocked long-double oracle; no packing or shared GEMM code.
long double reference(const std::vector<float>& factor,
                      const std::vector<float>& original,
                      const std::vector<int>& permutation,
                      int n, int lda, bool cholesky)
{
    long double numerator = 0, denominator = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            long double product = 0;
            for (int k = 0; k < n; ++k) {
                const long double left = i < k ? 0.0L
                    : (!cholesky && i == k ? 1.0L : factor[k * lda + i]);
                const long double right = cholesky
                    ? (j < k ? 0.0L : factor[k * lda + j])
                    : (k > j ? 0.0L : factor[j * lda + k]);
                product += left * right;
            }
            const long double a = original[j * lda + permutation[i]];
            const long double difference = product - a;
            numerator += difference * difference;
            denominator += a * a;
        }
    }
    return std::sqrt(numerator / denominator);
}

void test_layouts()
{
    for (const int n : {1, 2, 3, 5, 11}) {
        for (const int padding : {0, 3}) {
            const int lda = n + padding;
            for (const bool cholesky : {false, true}) {
                for (const bool pivoting : {false, true}) {
                    if (cholesky && pivoting) continue;
                    std::vector<float> factor(static_cast<size_t>(lda) * n,
                        std::numeric_limits<float>::quiet_NaN());
                    std::vector<float> original(factor.size(), 0.0f);
                    std::vector<int> pivots(n), permutation(n);
                    std::iota(permutation.begin(), permutation.end(), 0);
                    for (int k = 0; k < n; ++k) pivots[k] = k + 1;
                    // A three-cycle detects using P^{-1} instead of P.
                    if (pivoting && n >= 3) {
                        pivots[0] = 3;
                        pivots[1] = 3;
                        for (int k = 0; k < n; ++k)
                            std::swap(permutation[k], permutation[pivots[k] - 1]);
                    }
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < n; ++i) {
                            if (!cholesky || i >= j)
                                factor[j * lda + i] = 0.13f * (i + 1)
                                    - 0.07f * (j + 1) + (i == j ? 2.0f : 0.0f);
                        }
                    }
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < n; ++i) {
                            long double product = 0;
                            for (int k = 0; k <= std::min(i, j); ++k) {
                                const long double l = !cholesky && i == k
                                    ? 1.0L : factor[k * lda + i];
                                const long double r = cholesky
                                    ? factor[k * lda + j] : factor[j * lda + k];
                                product += l * r;
                            }
                            original[j * lda + permutation[i]] = static_cast<float>(product);
                        }
                    }
                    const long double expected = reference(
                        factor, original, permutation, n, lda, cholesky);
                    for (const int tile : {1, 2, 3, 4, 512}) {
                        const double actual = cholesky
                            ? benchmark_residual::potrf_cpu_fp64_full(
                                factor.data(), original.data(), n, lda, tile)
                            : benchmark_residual::getrf_cpu_fp64_full(
                                factor.data(), original.data(), pivots.data(),
                                n, lda, pivoting, tile);
                        close_to(actual, expected, "layout/tile/permutation residual");
                    }
                }
            }
        }
    }
}

void test_precision_and_errors()
{
    // FP32 multiplication rounds (1+2^-23)^2 to 1+2^-22. A CPU FP64
    // residual must retain the lost 2^-46 term, rather than report zero.
    const float factor = std::nextafter(1.0f, 2.0f);
    const float original = factor * factor;
    const long double exact = static_cast<long double>(factor) * factor;
    close_to(benchmark_residual::potrf_cpu_fp64_full(&factor, &original, 1, 1),
        std::abs(exact - original) / original, "multiply before FP64 conversion");
    require(benchmark_residual::potrf_cpu_fp64_full(&factor, &original, 1, 1) > 0,
        "FP64 product must resolve FP32 rounding");

    const float zero = 0, one = 1;
    require(benchmark_residual::potrf_cpu_fp64_full(&zero, &zero, 1, 1) == 0,
        "zero numerator and denominator");
    require(std::isinf(benchmark_residual::potrf_cpu_fp64_full(&one, &zero, 1, 1)),
        "nonzero numerator with zero denominator");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    require(std::isinf(benchmark_residual::potrf_cpu_fp64_full(&nan, &one, 1, 1)),
        "nonfinite factor rejected");
    const int bad_pivot = 0;
    require(std::isinf(benchmark_residual::getrf_cpu_fp64_full(
        &one, &one, &bad_pivot, 1, 1, true)), "invalid pivot rejected");
    for (const int tile : {0, -1}) {
        bool thrown = false;
        try { benchmark_residual::potrf_cpu_fp64_full(&one, &one, 1, 1, tile); }
        catch (const std::invalid_argument&) { thrown = true; }
        require(thrown, "invalid tile rejected");
    }
    // Double accumulation would overflow if norms were formed as raw squares.
    const double large = 1.0e160;
    const double twice = 2.0e160;
    close_to(benchmark_residual::getrf_cpu_fp64_full(
        &large, &twice, nullptr, 1, 1, false), 0.5L, "scaled norm overflow");
    const double small = 1.0e-200;
    const double twice_small = 2.0e-200;
    close_to(benchmark_residual::getrf_cpu_fp64_full(
        &small, &twice_small, nullptr, 1, 1, false), 0.5L, "scaled norm underflow");
}

void test_metadata()
{
    unsigned int seed = 19;
    require(parse_benchmark_seed("0", seed) && seed == 0, "zero seed");
    require(parse_benchmark_seed("4294967295", seed) && seed == 4294967295u,
        "maximum seed");
    for (const char* text : {"", "-1", "+1", "1x", "4294967296"}) {
        seed = 19;
        require(!parse_benchmark_seed(text, seed) && seed == 19, "invalid seed");
    }
    require(!parse_benchmark_seed(nullptr, seed), "null seed");
    require(std::strcmp(residual_metric_name(ResidualMetric::CpuFp64RelativeResidual),
        "cpu_fp64_full_relative_residual") == 0, "distinct CSV metric");
    require(std::strcmp(residual_metric_limit_environment(
        ResidualMetric::CpuFp64RelativeResidual),
        "BENCHMARK_MAX_CPU_FP64_RELATIVE_RESIDUAL") == 0, "distinct threshold");
}
} // namespace

int main()
{
    try {
        test_layouts();
        test_precision_and_errors();
        test_metadata();
        std::cout << "PASS: " << checks << " checks; "
                  << benchmark_residual::cpu_fp64_backend_name() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
