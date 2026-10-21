option(HYGON_CPU_FP64_USE_CBLAS
    "Use a host CBLAS library for optional CPU FP64 residual validation" OFF)

function(configure_cpu_fp64_residual target_name)
    if(NOT HYGON_CPU_FP64_USE_CBLAS)
        return()
    endif()

    # This is a CPU BLAS dependency, not rocBLAS/hipBLAS. Fail explicitly if
    # requested but absent; never substitute a GPU DGEMM on FP32-only hardware.
    find_path(HYGON_CPU_CBLAS_INCLUDE_DIR NAMES cblas.h
        PATH_SUFFIXES openblas openblas-pthread openblas-openmp)
    find_library(HYGON_CPU_CBLAS_LIBRARY NAMES openblas cblas blas)
    if(NOT HYGON_CPU_CBLAS_INCLUDE_DIR OR NOT HYGON_CPU_CBLAS_LIBRARY)
        message(FATAL_ERROR
            "CPU FP64 validation requested with CBLAS, but host CBLAS was not found. "
            "Set HYGON_CPU_CBLAS_INCLUDE_DIR and HYGON_CPU_CBLAS_LIBRARY, "
            "or disable HYGON_CPU_FP64_USE_CBLAS for the slow portable backend.")
    endif()

    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_INCLUDES "${HYGON_CPU_CBLAS_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_LIBRARIES "${HYGON_CPU_CBLAS_LIBRARY}")
    check_cxx_source_compiles("
        #include <cblas.h>
        int main() {
            double a=1.0,b=1.0,c=0.0;
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
                        1,1,1,1.0,&a,1,&b,1,0.0,&c,1);
            return c == 1.0 ? 0 : 1;
        }" HYGON_CPU_CBLAS_LINKS)
    if(NOT HYGON_CPU_CBLAS_LINKS)
        message(FATAL_ERROR "Selected host BLAS does not link cblas_dgemm")
    endif()
    target_compile_definitions(${target_name} PRIVATE HYGON_CPU_FP64_USE_CBLAS)
    target_include_directories(${target_name} PRIVATE "${HYGON_CPU_CBLAS_INCLUDE_DIR}")
    target_link_libraries(${target_name} "${HYGON_CPU_CBLAS_LIBRARY}")
endfunction()
