add_executable(nvcr_rans_benchmark rans_benchmark.cpp)
target_link_libraries(nvcr_rans_benchmark PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_rans_benchmark)

add_executable(nvcr_fx_cpu_benchmark fx_cpu_benchmark.cpp)
target_include_directories(nvcr_fx_cpu_benchmark PRIVATE "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/tests")
target_link_libraries(nvcr_fx_cpu_benchmark PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_fx_cpu_benchmark)
