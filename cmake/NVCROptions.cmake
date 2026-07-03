option(NVCR_BUILD_TESTS "Build NVCR tests" ON)
option(NVCR_BUILD_BENCHMARKS "Build NVCR microbenchmarks" OFF)
option(NVCR_BUILD_EXAMPLES "Build NVCR examples" ON)
option(NVCR_BUILD_CLI "Build the native nvcr command-line tool" ON)
option(NVCR_BUILD_REFERENCE_TOOLS "Build the developer-only Python reference launcher" OFF)
option(NVCR_FETCH_DEPENDENCIES "Download missing lightweight dependencies" OFF)
option(NVCR_ENABLE_TENSORRT "Build the TensorRT/CUDA backend" OFF)
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
