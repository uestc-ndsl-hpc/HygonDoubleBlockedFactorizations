#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <cstdlib>
#include <iostream>

#include "utils/common/hip_error_check.h"
#include "utils/common/rocblas_compat.h"

#define ROCBLAS_CHECK(cmd)                                                 \
    do {                                                                   \
        rocblas_status status = cmd;                                       \
        if (status != rocblas_status_success) {                            \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__\
                      << ": " << rocblas_status_to_string(status)          \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#endif
