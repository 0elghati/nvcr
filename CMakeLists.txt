cmake_minimum_required(VERSION 3.24)

project(
    NVCR
    VERSION 0.3.0 # x-release-please-version
    DESCRIPTION "C++ and TensorRT runtime for the pinned DCVC-RT model profile"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(NVCROptions)
include(NVCRDependencies)
include(NVCRWarnings)
find_package(Threads REQUIRED)

# CMAKE_CUDA_ARCHITECTURES must be resolved and the CUDA language enabled
# before any target is created, otherwise targets snapshot an empty
# CUDA_ARCHITECTURES property. Auto-detect the CUDA compiler and the GPU
# architecture (Jetson or discrete) here so a plain
# `cmake -DNVCR_ENABLE_TENSORRT=ON` works without manual -D flags.
if(NVCR_ENABLE_TENSORRT)
    include(NVCRAutodetect)
    nvcr_autodetect_cuda()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(TensorRT REQUIRED)
endif()

add_subdirectory(third_party/dcvc_rans)


add_library(nvcr)
add_library(NVCR::nvcr ALIAS nvcr)

target_sources(
    nvcr
    PRIVATE
        src/bitstream/access_unit.cpp
        src/bitstream/packet_io.cpp
        src/common/error.cpp
        src/common/sha256.cpp
        src/configuration/configuration.cpp
        src/dcvcrt/rans_codec.cpp
        src/dcvcrt/runtime.cpp
        src/dcvcrt/sequence_state.cpp
        src/logging/logger.cpp
        src/memory/memory_pool.cpp
        src/runtime/frame.cpp
        src/runtime/packet.cpp
        src/runtime/runtime.cpp
        src/statistics/statistics.cpp
        $<TARGET_OBJECTS:nvcr_dcvc_rans>)

target_include_directories(
    nvcr
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_include_directories(
    nvcr SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dcvc_rans/src")

target_compile_features(nvcr PUBLIC cxx_std_20)
target_link_libraries(nvcr PRIVATE Threads::Threads)
set_target_properties(
    nvcr PROPERTIES
    CXX_EXTENSIONS OFF
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME nvcr)

nvcr_set_project_warnings(nvcr)

if(spdlog_FOUND)
    target_link_libraries(nvcr PRIVATE spdlog::spdlog)
    target_compile_definitions(nvcr PRIVATE NVCR_HAS_SPDLOG=1)
endif()

if(NVCR_ENABLE_TENSORRT)
    target_sources(
        nvcr
        PRIVATE
            src/dcvcrt/cuda_ops.cu
            src/dcvcrt/tensorrt_backend.cpp)
    target_link_libraries(nvcr PRIVATE CUDA::cudart TensorRT::TensorRT)
    target_compile_definitions(nvcr PRIVATE NVCR_HAS_TENSORRT=1)
    set_target_properties(
        nvcr PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
endif()

if(NVCR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NVCR_BUILD_CLI)
    if(NVCR_ENABLE_TENSORRT)
        add_subdirectory(cli)
    else()
        message(STATUS "NVCR CLI skipped because NVCR_ENABLE_TENSORRT=OFF")
    endif()
endif()

if(NVCR_BUILD_REFERENCE_TOOLS)
    add_subdirectory(tools)
endif()

if(NVCR_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()

if(NVCR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

install(
    TARGETS nvcr
    EXPORT NVCRTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
if(TARGET nvcr_cli)
    install(TARGETS nvcr_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
install(
    PROGRAMS scripts/nvcr_artifacts.py
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    RENAME nvcr-artifacts)
install(
    PROGRAMS
        scripts/prepare_dcvcrt_artifacts.sh
        scripts/build_dcvcrt_tensorrt.sh
        scripts/detect_platform.sh
        scripts/export_dcvcrt_onnx.py
        scripts/export_dcvcrt_p_onnx.py
        scripts/write_tensorrt_engine_manifest.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nvcr/scripts")
install(DIRECTORY configs/ DESTINATION "${CMAKE_INSTALL_DATADIR}/nvcr/configs")
install(
    FILES resources/nvcr.conf.example
    DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/nvcr")
install(
    FILES README.md ROADMAP.md CHANGELOG.md LICENSE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr")
install(
    DIRECTORY docs/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr/docs")
install(
    FILES
        third_party/dcvc_rans/LICENSE.MIT
        third_party/dcvc_rans/NOTICE.txt
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr/third_party/dcvc_rans")
install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp")
install(
    EXPORT NVCRTargets
    FILE NVCRTargets.cmake
    NAMESPACE NVCR::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NVCR")
install(
    FILES cmake/FindTensorRT.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NVCR")

configure_package_config_file(
    cmake/NVCRConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/NVCRConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NVCR")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/NVCRConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/NVCRConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/NVCRConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NVCR")
