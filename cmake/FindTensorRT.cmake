find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS
        "${TensorRT_ROOT}"
        "${TENSORRT_ROOT}"
        "$ENV{TensorRT_ROOT}"
        "$ENV{TENSORRT_ROOT}"
    PATH_SUFFIXES include)
find_library(TensorRT_NVINFER_LIBRARY nvinfer
    HINTS
        "${TensorRT_ROOT}"
        "${TENSORRT_ROOT}"
        "$ENV{TensorRT_ROOT}"
        "$ENV{TENSORRT_ROOT}"
    PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY)

if(TensorRT_FOUND AND NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(
        TensorRT::TensorRT PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()

