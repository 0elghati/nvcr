# Cross-compilation toolchain for the NVCR Jetson Orin/L4T 36.4 package.
#
# The x86_64 build host must provide:
#   NVCR_JETSON_SYSROOT      extracted JetPack/L4T target root filesystem
#   NVCR_JETSON_CROSS_PREFIX compiler prefix, default aarch64-linux-gnu-
#   NVCR_CUDA_CROSS_ROOT     CUDA toolkit containing aarch64-linux targets
#
# TensorRT headers and libraries must be present below the target sysroot.
# Host tools are deliberately kept outside the sysroot so CMake can execute
# code generators during the cross build while target binaries are never run.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT DEFINED NVCR_JETSON_SYSROOT OR NVCR_JETSON_SYSROOT STREQUAL "")
    if(DEFINED ENV{NVCR_JETSON_SYSROOT})
        set(NVCR_JETSON_SYSROOT "$ENV{NVCR_JETSON_SYSROOT}")
    else()
        message(FATAL_ERROR "NVCR_JETSON_SYSROOT must point to the JetPack/L4T target rootfs")
    endif()
endif()
file(REAL_PATH "${NVCR_JETSON_SYSROOT}" NVCR_JETSON_SYSROOT)
if(NOT IS_DIRECTORY "${NVCR_JETSON_SYSROOT}")
    message(FATAL_ERROR "NVCR_JETSON_SYSROOT is not a directory: ${NVCR_JETSON_SYSROOT}")
endif()

if(NOT DEFINED NVCR_JETSON_CROSS_PREFIX OR NVCR_JETSON_CROSS_PREFIX STREQUAL "")
    if(DEFINED ENV{NVCR_JETSON_CROSS_PREFIX} AND NOT "$ENV{NVCR_JETSON_CROSS_PREFIX}" STREQUAL "")
        set(NVCR_JETSON_CROSS_PREFIX "$ENV{NVCR_JETSON_CROSS_PREFIX}")
    else()
        set(NVCR_JETSON_CROSS_PREFIX "aarch64-linux-gnu-")
    endif()
endif()

set(CMAKE_C_COMPILER "${NVCR_JETSON_CROSS_PREFIX}gcc" CACHE FILEPATH "Jetson AArch64 C compiler")
set(CMAKE_CXX_COMPILER "${NVCR_JETSON_CROSS_PREFIX}g++" CACHE FILEPATH "Jetson AArch64 C++ compiler")
set(CMAKE_ASM_COMPILER "${NVCR_JETSON_CROSS_PREFIX}gcc" CACHE FILEPATH "Jetson AArch64 assembler")
set(CMAKE_SYSROOT "${NVCR_JETSON_SYSROOT}" CACHE PATH "Jetson target sysroot")

if(NOT DEFINED NVCR_CUDA_CROSS_ROOT OR NVCR_CUDA_CROSS_ROOT STREQUAL "")
    if(DEFINED ENV{NVCR_CUDA_CROSS_ROOT} AND NOT "$ENV{NVCR_CUDA_CROSS_ROOT}" STREQUAL "")
        set(NVCR_CUDA_CROSS_ROOT "$ENV{NVCR_CUDA_CROSS_ROOT}")
    else()
        set(NVCR_CUDA_CROSS_ROOT "/usr/local/cuda")
    endif()
endif()
if(NOT IS_DIRECTORY "${NVCR_CUDA_CROSS_ROOT}")
    message(FATAL_ERROR "NVCR_CUDA_CROSS_ROOT is not a directory: ${NVCR_CUDA_CROSS_ROOT}")
endif()

set(CMAKE_CUDA_COMPILER "${NVCR_CUDA_CROSS_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA cross compiler")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "AArch64 compiler used by nvcc")
set(CMAKE_CUDA_COMPILER_TARGET "aarch64-linux" CACHE STRING "CUDA target triple")
set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "Jetson Orin CUDA architecture")
set(CMAKE_CUDA_FLAGS_INIT "--target-dir aarch64-linux --sysroot=${NVCR_JETSON_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH
    "${NVCR_JETSON_SYSROOT}"
    "${NVCR_CUDA_CROSS_ROOT}/targets/aarch64-linux")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_EXE_LINKER_FLAGS_INIT "--sysroot=${NVCR_JETSON_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "--sysroot=${NVCR_JETSON_SYSROOT}")

