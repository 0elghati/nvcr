add_executable(dcvcrt_reference nvcr.cpp)
set_target_properties(dcvcrt_reference PROPERTIES OUTPUT_NAME dcvcrt-reference)
target_compile_features(dcvcrt_reference PRIVATE cxx_std_20)
target_compile_definitions(
    dcvcrt_reference PRIVATE NVCR_DEFAULT_DCVCRT_ROOT="${NVCR_DCVCRT_ROOT}")
nvcr_set_project_warnings(dcvcrt_reference)

