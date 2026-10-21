#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifdef HYGON_CPU_FP64_USE_CBLAS
#include <cblas.h>
#endif

namespace benchmark_residual {

inline const char* cpu_fp64_backend_name()
{
#ifdef HYGON_CPU_FP64_USE_CBLAS
    return "CPU CBLAS DGEMM";
#else
    return "portable CPU FP64 (single-threaded)";
#endif
}

namespace detail {

// Scaled sum of squares avoids overflow/underflow in the norm accumulation.
struct SumOfSquares {
    double scale = 0.0;
    double sum = 1.0;
    bool finite = true;

    void add(double value)
    {
        const double magnitude = std::abs(value);
        if (!std::isfinite(magnitude)) {
            finite = false;
        } else if (magnitude != 0.0) {
            if (scale < magnitude) {
                const double ratio = scale / magnitude;
                sum = 1.0 + sum * ratio * ratio;
                scale = magnitude;
            } else {
                const double ratio = magnitude / scale;
                sum += ratio * ratio;
            }
        }
    }
};

inline void multiply_accumulate(int m, int n, int k,
                                const double* left,
                                const double* right,
                                double* product)
{
#ifdef HYGON_CPU_FP64_USE_CBLAS
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, left, m, right, k, 1.0, product, m);
#else
    for (int column = 0; column < n; ++column) {
        for (int inner = 0; inner < k; ++inner) {
            const double value = right[static_cast<size_t>(column) * k + inner];
            for (int row = 0; row < m; ++row) {
                product[static_cast<size_t>(column) * m + row]
                    += left[static_cast<size_t>(inner) * m + row] * value;
            }
        }
    }
#endif
}

template <typename LeftValue, typename RightValue, typename OriginalValue>
double full_residual(int n, int tile,
                     LeftValue left_value,
                     RightValue right_value,
                     OriginalValue original_value)
{
    if (n <= 0 || tile <= 0) {
        throw std::invalid_argument("CPU residual order and tile must be positive");
    }
    tile = std::min(tile, n);
    const size_t capacity = static_cast<size_t>(tile) * tile;
    std::vector<double> left(capacity);
    std::vector<double> right(capacity);
    std::vector<double> product(capacity);
    SumOfSquares residual_norm;
    SumOfSquares original_norm;

    // Each output tile covers the full reduction. No GPU product is reused.
    // Both LU and LL^T have zero products for k > min(row, column).
    for (int column = 0; column < n;) {
        const int columns = std::min(tile, n - column);
        for (int row = 0; row < n;) {
            const int rows = std::min(tile, n - row);
            std::fill(product.begin(), product.begin()
                          + static_cast<size_t>(rows) * columns, 0.0);
            const int reduction_end = std::min(row + rows, column + columns);
            for (int inner = 0; inner < reduction_end;) {
                const int count = std::min(tile, reduction_end - inner);
                for (int k = 0; k < count; ++k) {
                    for (int i = 0; i < rows; ++i) {
                        left[static_cast<size_t>(k) * rows + i]
                            = left_value(row + i, inner + k);
                    }
                }
                for (int j = 0; j < columns; ++j) {
                    for (int k = 0; k < count; ++k) {
                        right[static_cast<size_t>(j) * count + k]
                            = right_value(inner + k, column + j);
                    }
                }
                multiply_accumulate(rows, columns, count,
                                    left.data(), right.data(), product.data());
                inner += count;
            }
            for (int j = 0; j < columns; ++j) {
                for (int i = 0; i < rows; ++i) {
                    const double original = original_value(row + i, column + j);
                    original_norm.add(original);
                    residual_norm.add(product[static_cast<size_t>(j) * rows + i]
                                      - original);
                }
            }
            row += rows;
        }
        column += columns;
    }

    if (!residual_norm.finite || !original_norm.finite) {
        return std::numeric_limits<double>::infinity();
    }
    if (original_norm.scale == 0.0) {
        return residual_norm.scale == 0.0
            ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return (residual_norm.scale / original_norm.scale)
        * std::sqrt(residual_norm.sum / original_norm.sum);
}

} // namespace detail

// All arrays are host-resident and column-major. Their stored values are
// converted to double before multiplication, not after a lower-precision GEMM.
template <typename T>
double getrf_cpu_fp64_full(const T* packed_lu, const T* original,
                          const int* pivots, int n, int lda,
                          bool pivoting, int tile = 512)
{
    if (n <= 0 || lda < n || tile <= 0 || packed_lu == nullptr
        || original == nullptr || (pivoting && pivots == nullptr)) {
        throw std::invalid_argument("Invalid CPU LU residual inputs");
    }
    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    if (pivoting) {
        for (int row = 0; row < n; ++row) {
            // GETRF supplies one-based global row indices.
            if (pivots[row] < row + 1 || pivots[row] > n) {
                return std::numeric_limits<double>::infinity();
            }
            std::swap(permutation[row], permutation[pivots[row] - 1]);
        }
    }
    return detail::full_residual(n, tile,
        [&](int row, int k) -> double {
            if (row < k) return 0.0;
            if (row == k) return 1.0;
            return static_cast<double>(packed_lu[static_cast<size_t>(k) * lda + row]);
        },
        [&](int k, int column) -> double {
            return k > column ? 0.0
                : static_cast<double>(packed_lu[static_cast<size_t>(column) * lda + k]);
        },
        [&](int row, int column) -> double {
            return static_cast<double>(original[static_cast<size_t>(column) * lda
                                                + permutation[row]]);
        });
}

template <typename T>
double potrf_cpu_fp64_full(const T* packed_lower, const T* original,
                          int n, int lda, int tile = 512)
{
    if (n <= 0 || lda < n || tile <= 0 || packed_lower == nullptr
        || original == nullptr) {
        throw std::invalid_argument("Invalid CPU Cholesky residual inputs");
    }
    return detail::full_residual(n, tile,
        [&](int row, int k) -> double {
            return row < k ? 0.0
                : static_cast<double>(packed_lower[static_cast<size_t>(k) * lda + row]);
        },
        [&](int k, int column) -> double {
            return column < k ? 0.0
                : static_cast<double>(packed_lower[static_cast<size_t>(k) * lda + column]);
        },
        [&](int row, int column) -> double {
            return static_cast<double>(original[static_cast<size_t>(column) * lda + row]);
        });
}

} // namespace benchmark_residual
