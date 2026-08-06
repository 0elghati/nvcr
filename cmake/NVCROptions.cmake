option(NVCR_BUILD_TESTS "Build NVCR tests" ON)
option(NVCR_BUILD_BENCHMARKS "Build NVCR microbenchmarks" OFF)
option(NVCR_BUILD_EXAMPLES "Build NVCR examples" ON)
option(NVCR_BUILD_CLI "Build the native nvcr command-line tool" ON)
option(NVCR_BUILD_REFERENCE_TOOLS "Build the developer-only Python reference launcher" OFF)
option(NVCR_FETCH_DEPENDENCIES "Download missing lightweight dependencies" OFF)
option(NVCR_ENABLE_TENSORRT "Build the TensorRT/CUDA backend" OFF)
# NVCR_ENABLE_CUDA is currently implied by NVCR_ENABLE_TENSORRT. It remains an
# explicit option so CUDA support can be separated from TensorRT later.
option(NVCR_ENABLE_CUDA "Enable CUDA support (implied when NVCR_ENABLE_TENSORRT is ON)" OFF)
# The current umbrella target includes the DCVC-RT adapter when this is enabled.
option(NVCR_ENABLE_DCVC_RT "Enable the DCVC-RT codec adapter" ON)
# Reserved for the future stream/manifest inspection tool.
option(NVCR_ENABLE_TOOLS "Build stream and manifest inspection tools" OFF)
# Enables libFuzzer targets under tests/fuzz/.
option(NVCR_ENABLE_FUZZING "Build libFuzzer fuzz targets" OFF)
option(NVCR_ENABLE_OPENCV "Enable optional OpenCV frame interoperability" OFF)
option(NVCR_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NVCR_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(
    NVCR_DCVCRT_ROOT
    ""
    CACHE PATH
    "Default path to a working upstream DCVC-RT checkout for the developer reference tool")

set(
    NVCR_TENSORRT_ENGINE_DIR
    ""
    CACHE PATH
    "Directory containing built DCVC-RT TensorRT plans for integration tests")

set(
    NVCR_TENSORRT_ENGINE_DIRS
    ""
    CACHE STRING
    "Semicolon-separated additional DCVC-RT TensorRT engine directories for profile-matrix tests")

set(
    NVCR_DCVCRT_720P_GOLDEN_INPUT
    ""
    CACHE FILEPATH
    "FourPeople 1280x720 YUV420P8 source used by the pinned I-frame golden test")

set(NVCR_CUDA_ARCH_SET "auto" CACHE STRING
    "CUDA architecture selection when CMAKE_CUDA_ARCHITECTURES is not set explicitly: \
'auto' detects this build machine's GPU (fastest build, single-GPU dev/local installs); \
'portable' builds a redistributable fat binary covering common Jetson and discrete \
RTX/datacenter GPU architectures (slower build, for release packaging). Pass an explicit \
-DCMAKE_CUDA_ARCHITECTURES=... to target something else entirely.")
set_property(CACHE NVCR_CUDA_ARCH_SET PROPERTY STRINGS auto portable)
