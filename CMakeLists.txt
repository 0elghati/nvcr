cmake_minimum_required(VERSION 3.24)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" NVCR_VERSION LIMIT_COUNT 1)
string(STRIP "${NVCR_VERSION}" NVCR_VERSION)
if(NOT NVCR_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "version.txt must contain MAJOR.MINOR.PATCH")
endif()

project(
    NVCR
    VERSION "${NVCR_VERSION}"
    DESCRIPTION "Native runtime architecture for neural video codecs"
    LANGUAGES CXX)

set(NVCR_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY "${NVCR_GENERATED_INCLUDE_DIR}/nvcr/common")
configure_file(
    include/nvcr/common/versions.hpp.in
    "${NVCR_GENERATED_INCLUDE_DIR}/nvcr/common/versions.hpp"
    @ONLY)

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
if(NVCR_ENABLE_TENSORRT OR NVCR_ENABLE_DCVC_UF)
    include(NVCRAutodetect)
    nvcr_autodetect_cuda()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()
if(NVCR_ENABLE_TENSORRT)
    find_package(TensorRT REQUIRED)
endif()
if(NVCR_ENABLE_ONNXRUNTIME)
    find_package(OnnxRuntime REQUIRED)
endif()

add_subdirectory(third_party/dcvc_rans)
add_subdirectory(third_party/msrtc_rans)
add_subdirectory(third_party/dcvcuf_rans)

# DCVC-UF entropy is codec-local. It always builds so its CPU conformance test
# runs without CUDA, but it enters libnvcr only with NVCR_ENABLE_DCVC_UF.
# The test-only diagnostics observer travels with it: the coder reports entropy
# indexes to it on both execution paths.
add_library(nvcr_dcvcuf_entropy OBJECT
    src/dcvcuf/entropy/uf_rans_codec.cpp
    src/dcvcuf/diagnostics.cpp)
target_include_directories(
    nvcr_dcvcuf_entropy PRIVATE
        "${NVCR_GENERATED_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf")
target_include_directories(
    nvcr_dcvcuf_entropy SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dcvcuf_rans/src")
target_compile_features(nvcr_dcvcuf_entropy PRIVATE cxx_std_20)
set_target_properties(
    nvcr_dcvcuf_entropy PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
nvcr_set_project_warnings(nvcr_dcvcuf_entropy)

# ---------------------------------------------------------------------------
# Module-boundary interface targets
#
# These INTERFACE targets carry no sources; they define the dependency rules
# that enforce the codec-adapter / execution-provider / runtime-core separation
# described in docs/architecture.md. Physical source movement into separate
# object-library targets remains an implementation detail.
#
# nvcr_base          – public types: Error/Result, Frame, Packet, Timestamp,
#                      PixelFormat, MemoryKind, logging, statistics, memory pool
# nvcr_format        – bitstream/access-unit framing (nvcr/bitstream/*)
# nvcr_codec_api     – abstract codec session interface (nvcr/codec/*)
# nvcr_provider_api  – abstract execution-provider interface (nvcr/provider/*)
# ---------------------------------------------------------------------------
add_library(nvcr_base INTERFACE)
add_library(NVCR::base ALIAS nvcr_base)
target_include_directories(
    nvcr_base INTERFACE
        "$<BUILD_INTERFACE:${NVCR_GENERATED_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(nvcr_base INTERFACE cxx_std_20)

add_library(nvcr_format INTERFACE)
add_library(NVCR::format ALIAS nvcr_format)
target_link_libraries(nvcr_format INTERFACE nvcr_base)

add_library(nvcr_codec_api INTERFACE)
add_library(NVCR::codec_api ALIAS nvcr_codec_api)
target_link_libraries(nvcr_codec_api INTERFACE nvcr_base)

add_library(nvcr_provider_api INTERFACE)
add_library(NVCR::provider_api ALIAS nvcr_provider_api)
target_link_libraries(nvcr_provider_api INTERFACE nvcr_base)

add_library(nvcr)
add_library(NVCR::nvcr ALIAS nvcr)

target_sources(
    nvcr
    PRIVATE
        src/bitstream/access_unit.cpp
        src/bitstream/numerics_identity.cpp
        src/bitstream/packet_io.cpp
        src/artifacts/catalog.cpp
        src/artifacts/resolver.cpp
        src/codec/sequence_state.cpp
        src/codec/runtime.cpp
        src/codec/stream_numerics.cpp
        src/common/error.cpp
        src/common/json.cpp
        src/common/sha256.cpp
        src/configuration/configuration.cpp
        src/dcvcrt/adapter.cpp
        src/dcvcrt/fx_backend.cpp
        src/dcvcrt/host_backend.cpp
        src/dcvcrt/host_ops.cpp
        src/dcvcrt/numerics.cpp
        src/dcvcrt/orchestration.cpp
        src/dcvcrt/payload.cpp
        src/dcvcrt/rans_codec.cpp
        src/dcvcrt/registration.cpp
        src/dcvcrt/session.cpp
        src/logging/logger.cpp
        src/memory/memory_pool.cpp
        src/mlvc/bundle.cpp
        src/mlvc/entropy.cpp
        src/mlvc/host_ops.cpp
        src/mlvc/record.cpp
        src/mlvc/registration.cpp
        src/mlvc/session.cpp
        src/runtime/frame.cpp
        src/runtime/packet.cpp
        src/runtime/registry.cpp
        src/runtime/runtime.cpp
        src/statistics/statistics.cpp
        src/deterministic/fx.cpp
        src/deterministic/fx_program.cpp
        src/deterministic/normative_stage.cpp
        src/deterministic/program.cpp
        $<TARGET_OBJECTS:nvcr_dcvc_rans>
        $<TARGET_OBJECTS:nvcr_msrtc_rans>)

# Exact numerics profiles (issue #216) must not be contracted or
# reassociated: the program format fixes every rounding step so all targets
# agree bit for bit.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        src/deterministic/program.cpp src/deterministic/normative_stage.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off;-fno-fast-math")
endif()

target_include_directories(
    nvcr
    PUBLIC
        "$<BUILD_INTERFACE:${NVCR_GENERATED_INCLUDE_DIR}>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_include_directories(
    nvcr SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dcvc_rans/src")
target_include_directories(
    nvcr SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/msrtc_rans/include")

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
            src/provider/tensorrt/session.cpp
            src/dcvcrt/backend/tensorrt/cuda_ops.cu
            src/dcvcrt/backend/tensorrt/engine_specs.cpp
            src/dcvcrt/backend/tensorrt/backend.cpp
            src/provider/tensorrt/fx/kernel_table.cpp
            src/provider/tensorrt/fx/ops.cu
            src/provider/tensorrt/fx/plans.cpp
            src/provider/tensorrt/fx/plugins.cpp)
    target_link_libraries(nvcr PRIVATE CUDA::cudart CUDA::cuda_driver TensorRT::TensorRT)
    target_compile_definitions(nvcr PRIVATE NVCR_HAS_TENSORRT=1)

    # nvcr.fx.v1 on TensorRT: Triton kernels compiled ahead of time for the build
    # GPU's architecture. Without NVCR_TENSORRT_FX_PYTHON (a Python with Triton
    # 3.5.1) the table is empty and the provider does not claim the profile.
    set(NVCR_TENSORRT_FX_PYTHON "" CACHE FILEPATH "Python with Triton that compiles the nvcr.fx.v1 kernels")
    set(nvcr_fx_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/provider/tensorrt/fx")
    set(nvcr_fx_cubins "${CMAKE_CURRENT_BINARY_DIR}/generated/fx_cubins.cpp")
    if(NVCR_TENSORRT_FX_PYTHON)
        list(GET CMAKE_CUDA_ARCHITECTURES 0 nvcr_fx_sm)
        string(REGEX REPLACE "[^0-9].*$" "" nvcr_fx_sm "${nvcr_fx_sm}")
        add_custom_command(
            OUTPUT "${nvcr_fx_cubins}"
            COMMAND "${NVCR_TENSORRT_FX_PYTHON}" "${nvcr_fx_dir}/kernels/aot_compile.py"
                --variants "${CMAKE_CURRENT_SOURCE_DIR}/configs/tensorrt/fx-kernel-variants.json"
                --out "${nvcr_fx_cubins}" --header "${nvcr_fx_dir}/kernel_table.hpp"
                --target "cuda:${nvcr_fx_sm}:32"
            DEPENDS "${nvcr_fx_dir}/kernels/aot_compile.py" "${nvcr_fx_dir}/kernels/fx_kernels.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/configs/tensorrt/fx-kernel-variants.json"
            VERBATIM)
    else()
        file(WRITE "${nvcr_fx_cubins}"
            "#include \"${nvcr_fx_dir}/kernel_table.hpp\"\n"
            "namespace nvcr::tensorrt::fx {\n"
            "const KernelEntry kernel_entries[] = {{\"\", nullptr, 0, \"\", 0, 0}};\n"
            "const std::size_t kernel_entry_count = 0;\n"
            "const char kernel_target[] = \"\";\n"
            "const char kernel_triton_version[] = \"\";\n"
            "}  // namespace nvcr::tensorrt::fx\n")
    endif()
    set_source_files_properties("${nvcr_fx_cubins}" PROPERTIES COMPILE_OPTIONS "-O0")
    target_sources(nvcr PRIVATE "${nvcr_fx_cubins}")

    add_executable(nvcr_fx_plans src/provider/tensorrt/fx/build_plans_main.cpp)
    set_target_properties(nvcr_fx_plans PROPERTIES OUTPUT_NAME nvcr-fx-plans)
    target_link_libraries(nvcr_fx_plans PRIVATE nvcr CUDA::cudart TensorRT::TensorRT)

    # Exact executor kernels. Precise division, no flush-to-zero, and no
    # compiler-formed FMAs: results must equal the CPU reference bit for bit.
    add_library(nvcr_deterministic_cuda OBJECT src/deterministic/cuda_executor.cu)
    target_include_directories(nvcr_deterministic_cuda PRIVATE
        "${NVCR_GENERATED_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_options(nvcr_deterministic_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-ftz=false;-prec-div=true;-prec-sqrt=true;-fmad=false>)
    target_link_libraries(nvcr_deterministic_cuda PRIVATE CUDA::cudart)
    set_target_properties(nvcr_deterministic_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
    target_sources(nvcr PRIVATE $<TARGET_OBJECTS:nvcr_deterministic_cuda>)
    set_target_properties(
        nvcr PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
endif()

if(NVCR_ENABLE_ONNXRUNTIME)
    target_sources(nvcr PRIVATE src/provider/onnxruntime/session.cpp)
    target_link_libraries(nvcr PRIVATE OnnxRuntime::OnnxRuntime)
    target_compile_definitions(nvcr PRIVATE NVCR_HAS_ONNXRUNTIME=1)
endif()

if(NVCR_ENABLE_TENSORRT OR NVCR_ENABLE_DCVC_UF)
    # DCVC-UF glue kernels for its TensorRT path, generated from pinned upstream
    # device code. Upstream's compile flags and half macros are part of the
    # arithmetic: they select fast-math intrinsics and route Half through float.
    # The flags mirror upstream setup.py plus torch's COMMON_NVCC_FLAGS.
    add_library(nvcr_dcvcuf_glue OBJECT src/dcvcuf/backend/tensorrt/glue.cu)
    target_compile_options(nvcr_dcvcuf_glue PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;--expt-relaxed-constexpr>
        # Upstream's kernels declare a few unused locals (warning 177).
        $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>)
    target_compile_definitions(nvcr_dcvcuf_glue PRIVATE
        __CUDA_NO_HALF_OPERATORS__
        __CUDA_NO_HALF_CONVERSIONS__
        __CUDA_NO_BFLOAT16_CONVERSIONS__
        __CUDA_NO_HALF2_OPERATORS__)
    set_target_properties(nvcr_dcvcuf_glue PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
endif()

if(NVCR_ENABLE_TENSORRT OR NVCR_ENABLE_DCVC_UF)
    # The DCVC-UF codec runs on the TensorRT provider, the native fused path, or both.
    target_sources(nvcr PRIVATE
        src/dcvcuf/adapter.cpp
        src/dcvcuf/registration.cpp
        $<TARGET_OBJECTS:nvcr_dcvcuf_entropy>
        $<TARGET_OBJECTS:nvcr_dcvcuf_rans>)
    target_include_directories(nvcr PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/entropy")
    target_compile_definitions(nvcr PRIVATE NVCR_HAS_DCVC_UF=1)
    if(NVCR_ENABLE_TENSORRT)
        target_sources(nvcr PRIVATE
            src/dcvcuf/tensorrt_session.cpp
            src/dcvcuf/backend/tensorrt/frames.cu
            $<TARGET_OBJECTS:nvcr_dcvcuf_glue>)
    endif()
endif()

if(NVCR_ENABLE_DCVC_UF)
    if(NOT IS_DIRECTORY "${NVCR_DCVCUF_SOURCE_ROOT}/src/layers/extensions/inference")
        message(FATAL_ERROR
            "NVCR_ENABLE_DCVC_UF requires NVCR_DCVCUF_SOURCE_ROOT to name the pinned DCVC-UF checkout")
    endif()
    if(NOT IS_DIRECTORY "${NVCR_DCVCUF_CUTLASS_ROOT}/include/cutlass")
        message(FATAL_ERROR
            "NVCR_ENABLE_DCVC_UF requires NVCR_DCVCUF_CUTLASS_ROOT to name a CUTLASS checkout")
    endif()
    list(LENGTH CMAKE_CUDA_ARCHITECTURES NVCR_DCVCUF_ARCH_COUNT)
    if(NOT NVCR_DCVCUF_ARCH_COUNT EQUAL 1 OR
       NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "DCVC-UF fused kernels require one numeric CMAKE_CUDA_ARCHITECTURES target")
    endif()
    set(NVCR_DCVCUF_TARGET_SM "${CMAKE_CUDA_ARCHITECTURES}")

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_package(Torch REQUIRED)
    set(NVCR_DCVCUF_GENERATED_SOURCE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/dcvcuf_upstream")
    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/backends/dcvcuf/materialize_native_sources.py"
            --reference-root "${NVCR_DCVCUF_SOURCE_ROOT}"
            --output "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}"
        RESULT_VARIABLE NVCR_DCVCUF_MATERIALIZE_RESULT
        ERROR_VARIABLE NVCR_DCVCUF_MATERIALIZE_ERROR)
    if(NOT NVCR_DCVCUF_MATERIALIZE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to materialize Python-free DCVC-UF sources: ${NVCR_DCVCUF_MATERIALIZE_ERROR}")
    endif()

    file(GLOB NVCR_DCVCUF_CUDA_SOURCES CONFIGURE_DEPENDS
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/cutlass/*.cu"
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/elementwise/*.cu")
    add_library(nvcr_dcvcuf_upstream OBJECT
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/dmc_common.cpp"
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/dmci_proxy.cpp"
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/dmc_hts_proxy.cpp"
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/layers_proxy.cpp"
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}/memory_pool.cpp"
        src/dcvcuf/backend/upstream/native_rans.cpp
        ${NVCR_DCVCUF_CUDA_SOURCES})
    target_include_directories(nvcr_dcvcuf_upstream PRIVATE
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/backend/upstream"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/entropy"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${NVCR_GENERATED_INCLUDE_DIR}"
        "${NVCR_DCVCUF_CUTLASS_ROOT}/include"
        "${NVCR_DCVCUF_CUTLASS_ROOT}/tools/util/include"
        ${TORCH_INCLUDE_DIRS})
    target_compile_features(nvcr_dcvcuf_upstream PRIVATE cxx_std_20)
    target_link_libraries(nvcr_dcvcuf_upstream PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
    target_compile_options(nvcr_dcvcuf_upstream PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wno-deprecated-declarations>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--extra-device-vectorization;--split-compile=2;-Wno-deprecated-declarations>)
    target_compile_definitions(nvcr_dcvcuf_upstream PRIVATE
        CURRENT_DEVICE_SM=${NVCR_DCVCUF_TARGET_SM}
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_BFLOAT16_CONVERSIONS__>
        $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF2_OPERATORS__>)
    set_target_properties(nvcr_dcvcuf_upstream PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)

    target_sources(nvcr PRIVATE
        src/dcvcuf/native_session.cpp
        $<TARGET_OBJECTS:nvcr_dcvcuf_upstream>)
    target_include_directories(nvcr PRIVATE
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/backend/upstream")
    target_link_libraries(nvcr PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
    target_compile_definitions(nvcr PRIVATE NVCR_HAS_DCVCUF_NATIVE=1)

    # Bitwise parity of NVCR's DCVC-UF glue port with the pinned upstream kernels.
    add_executable(nvcr_dcvcuf_glue_parity_tests
        tests/dcvcuf_glue_parity_tests.cpp
        $<TARGET_OBJECTS:nvcr_dcvcuf_glue>)
    target_include_directories(nvcr_dcvcuf_glue_parity_tests PRIVATE
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/backend/tensorrt")
    target_link_libraries(nvcr_dcvcuf_glue_parity_tests PRIVATE
        nvcr ${TORCH_LIBRARIES} CUDA::cudart)
    target_compile_features(nvcr_dcvcuf_glue_parity_tests PRIVATE cxx_std_20)

    add_executable(nvcr_dcvcuf_native_roundtrip
        tests/dcvcuf_native_roundtrip.cpp)
    target_include_directories(nvcr_dcvcuf_native_roundtrip PRIVATE
        "${NVCR_DCVCUF_GENERATED_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/backend/upstream"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/dcvcuf/entropy")
    target_link_libraries(nvcr_dcvcuf_native_roundtrip PRIVATE
        nvcr ${TORCH_LIBRARIES} CUDA::cudart)
    target_compile_features(nvcr_dcvcuf_native_roundtrip PRIVATE cxx_std_20)
endif()

if(NVCR_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NVCR_BUILD_CLI)
    add_subdirectory(cli)
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
    TARGETS nvcr nvcr_base nvcr_format nvcr_codec_api nvcr_provider_api
    EXPORT NVCRTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
if(TARGET nvcr_cli)
    install(TARGETS nvcr_cli RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
if(TARGET nvcr_fx_plans)
    install(TARGETS nvcr_fx_plans RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/bin")
configure_file(
    scripts/nvcr_artifacts.py
    "${CMAKE_CURRENT_BINARY_DIR}/generated/bin/nvcr-artifacts"
    @ONLY)
install(
    PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/generated/bin/nvcr-artifacts"
    DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(
    FILES scripts/nvcr_device.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nvcr/scripts")
install(
    PROGRAMS
        scripts/detect_platform.sh
        scripts/install.sh
        scripts/install_from_source.sh
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nvcr/scripts")
install(
    PROGRAMS
        scripts/backends/dcvcrt/prepare_artifacts.sh
        scripts/backends/dcvcrt/build_tensorrt.sh
        scripts/backends/dcvcrt/export_i_onnx.py
        scripts/backends/dcvcrt/export_p_onnx.py
        scripts/backends/dcvcrt/write_tensorrt_engine_manifest.py
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nvcr/scripts/backends/dcvcrt")
install(DIRECTORY configs/ DESTINATION "${CMAKE_INSTALL_DATADIR}/nvcr/configs")
install(
    FILES resources/nvcr.conf.example
    DESTINATION "${CMAKE_INSTALL_SYSCONFDIR}/nvcr")
install(
    FILES README.md ROADMAP.md CHANGELOG.md LICENSE
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr")
install(
    FILES
        THIRD_PARTY_NOTICES.md
        MODEL_LICENSES.md
        ASSET_DISTRIBUTION_POLICY.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr")
install(
    DIRECTORY docs/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr/docs"
    PATTERN "*.orig" EXCLUDE
    PATTERN "*.rej" EXCLUDE
    PATTERN "*.bak" EXCLUDE)
install(
    FILES
        third_party/dcvc_rans/LICENSE.MIT
        third_party/dcvc_rans/NOTICE.txt
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr/third_party/dcvc_rans")
install(
    FILES
        third_party/dcvcuf_rans/LICENSE.txt
        third_party/dcvcuf_rans/NOTICE.txt
        third_party/dcvcuf_rans/UPSTREAM.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr/third_party/dcvcuf_rans")
install(
    DIRECTORY third_party/dcvc_rt/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/nvcr/third_party/dcvc_rt")
install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp")
install(
    FILES "${NVCR_GENERATED_INCLUDE_DIR}/nvcr/common/versions.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nvcr/common")
install(
    EXPORT NVCRTargets
    FILE NVCRTargets.cmake
    NAMESPACE NVCR::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/NVCR")
install(
    FILES cmake/FindTensorRT.cmake cmake/FindOnnxRuntime.cmake
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
