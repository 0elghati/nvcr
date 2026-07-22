# When no explicit TensorRT_ROOT is given, try to discover a pip-installed
# `tensorrt` Python package location. This covers environments (including some
# Jetson JetPack images) where TensorRT ships only as a Python wheel.
if(NOT TensorRT_ROOT AND NOT TENSORRT_ROOT AND NOT DEFINED ENV{TensorRT_ROOT} AND NOT DEFINED ENV{TENSORRT_ROOT})
    find_program(_nvcr_python3 python3)
    if(_nvcr_python3)
        execute_process(
            COMMAND "${_nvcr_python3}" -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))"
            OUTPUT_VARIABLE _nvcr_trt_pkg_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _nvcr_trt_pkg_result)
        if(_nvcr_trt_pkg_result EQUAL 0 AND _nvcr_trt_pkg_dir)
            set(TensorRT_ROOT "${_nvcr_trt_pkg_dir}")
        endif()
    endif()
endif()

# Debian/Ubuntu and JetPack .deb TensorRT installs put headers/libraries under
# plain system prefixes (/usr, /usr/local) or /usr/src/tensorrt; search those
# explicitly in addition to any user-provided TensorRT_ROOT/TENSORRT_ROOT.
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS
        "${TensorRT_ROOT}"
        "${TENSORRT_ROOT}"
        "$ENV{TensorRT_ROOT}"
        "$ENV{TENSORRT_ROOT}"
    PATHS
        /usr
        /usr/local
        /usr/src/tensorrt
        /opt/TensorRT
    PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
find_library(TensorRT_NVINFER_LIBRARY nvinfer
    HINTS
        "${TensorRT_ROOT}"
        "${TENSORRT_ROOT}"
        "$ENV{TensorRT_ROOT}"
        "$ENV{TENSORRT_ROOT}"
    PATHS
        /usr
        /usr/local
        /usr/src/tensorrt
        /opt/TensorRT
    PATH_SUFFIXES
        lib lib64
        lib/x86_64-linux-gnu lib/aarch64-linux-gnu
        targets/x86_64-linux/lib targets/aarch64-linux/lib)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _nvcr_trt_major_line REGEX "^#define NV_TENSORRT_MAJOR [0-9]+")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _nvcr_trt_minor_line REGEX "^#define NV_TENSORRT_MINOR [0-9]+")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _nvcr_trt_patch_line REGEX "^#define NV_TENSORRT_PATCH [0-9]+")
    if(_nvcr_trt_major_line AND _nvcr_trt_minor_line AND _nvcr_trt_patch_line)
        string(REGEX MATCH "NV_TENSORRT_MAJOR ([0-9]+)" _nvcr_trt_major_match "${_nvcr_trt_major_line}")
        string(REGEX MATCH "NV_TENSORRT_MINOR ([0-9]+)" _nvcr_trt_minor_match "${_nvcr_trt_minor_line}")
        string(REGEX MATCH "NV_TENSORRT_PATCH ([0-9]+)" _nvcr_trt_patch_match "${_nvcr_trt_patch_line}")
        string(REGEX REPLACE "NV_TENSORRT_MAJOR ([0-9]+)" "\\1" _nvcr_trt_major "${_nvcr_trt_major_match}")
        string(REGEX REPLACE "NV_TENSORRT_MINOR ([0-9]+)" "\\1" _nvcr_trt_minor "${_nvcr_trt_minor_match}")
        string(REGEX REPLACE "NV_TENSORRT_PATCH ([0-9]+)" "\\1" _nvcr_trt_patch "${_nvcr_trt_patch_match}")
        set(TensorRT_VERSION "${_nvcr_trt_major}.${_nvcr_trt_minor}.${_nvcr_trt_patch}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY
    VERSION_VAR TensorRT_VERSION)

if(TensorRT_FOUND AND NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(
        TensorRT::TensorRT PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()

