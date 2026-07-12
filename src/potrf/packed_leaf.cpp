#include "potrf/packed_leaf.h"

#include <cstddef>

#include "utils/common/error_check.h"

namespace {

template <typename T>
void copy_leaf(const T* source,
               int source_ld,
               T* destination,
               int destination_ld,
               int n,
               hipStream_t stream)
{
    if (n <= 0) {
        return;
    }

    HIP_CHECK(hipMemcpy2DAsync(
        destination,
        static_cast<size_t>(destination_ld) * sizeof(T),
        source,
        static_cast<size_t>(source_ld) * sizeof(T),
        static_cast<size_t>(n) * sizeof(T),
        n,
        hipMemcpyDeviceToDevice,
        stream));

    return;
}

} // namespace

extern "C" {

void pack_potrf_leaf(const float* source,
                     int source_ld,
                     float* destination,
                     int destination_ld,
                     int n,
                     hipStream_t stream)
{
    copy_leaf(
        source, source_ld, destination, destination_ld, n, stream);

    return;
}

void unpack_potrf_leaf(const float* source,
                       int source_ld,
                       float* destination,
                       int destination_ld,
                       int n,
                       hipStream_t stream)
{
    copy_leaf(
        source, source_ld, destination, destination_ld, n, stream);

    return;
}

void pack_potrf_leaf_double(const double* source,
                            int source_ld,
                            double* destination,
                            int destination_ld,
                            int n,
                            hipStream_t stream)
{
    copy_leaf(
        source, source_ld, destination, destination_ld, n, stream);

    return;
}

void unpack_potrf_leaf_double(const double* source,
                              int source_ld,
                              double* destination,
                              int destination_ld,
                              int n,
                              hipStream_t stream)
{
    copy_leaf(
        source, source_ld, destination, destination_ld, n, stream);

    return;
}

} // extern "C"
