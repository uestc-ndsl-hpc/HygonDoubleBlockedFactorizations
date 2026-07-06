set(ROCM_PATH /opt/dtk CACHE PATH "DTK/ROCm installation path")

if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(HIPCC_EXECUTABLE
        NAMES hipcc
        HINTS
            ${ROCM_PATH}/bin
            /opt/dtk/bin
            /opt/dtk-26.04/bin
            /opt/rocm/bin
            /public/software/compiler/rocm/dtk-24.04/bin
            /public/software/compiler/rocm/dtk-23.10/bin
    )

    if(HIPCC_EXECUTABLE)
        set(CMAKE_CXX_COMPILER ${HIPCC_EXECUTABLE} CACHE FILEPATH "CXX compiler" FORCE)
    endif()
endif()
