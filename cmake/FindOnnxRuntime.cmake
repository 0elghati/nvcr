# Locates an ONNX Runtime C/C++ release: set OnnxRuntime_ROOT (or the
# ONNXRUNTIME_ROOT variable or environment variable) to the extracted package.
find_path(OnnxRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
        "${OnnxRuntime_ROOT}"
        "${ONNXRUNTIME_ROOT}"
        "$ENV{OnnxRuntime_ROOT}"
        "$ENV{ONNXRUNTIME_ROOT}"
    PATHS
        /usr
        /usr/local
        /opt/onnxruntime
    PATH_SUFFIXES include include/onnxruntime)
find_library(OnnxRuntime_LIBRARY onnxruntime
    HINTS
        "${OnnxRuntime_ROOT}"
        "${ONNXRUNTIME_ROOT}"
        "$ENV{OnnxRuntime_ROOT}"
        "$ENV{ONNXRUNTIME_ROOT}"
    PATHS
        /usr
        /usr/local
        /opt/onnxruntime
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OnnxRuntime
    REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::OnnxRuntime)
    add_library(OnnxRuntime::OnnxRuntime UNKNOWN IMPORTED)
    set_target_properties(
        OnnxRuntime::OnnxRuntime PROPERTIES
        IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}")
endif()

mark_as_advanced(OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)
