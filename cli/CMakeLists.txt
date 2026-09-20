add_executable(nvcr_cli main.cpp)
set_target_properties(nvcr_cli PROPERTIES OUTPUT_NAME nvcr)
target_link_libraries(nvcr_cli PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_cli)

if(NVCR_ENABLE_TENSORRT)
    target_link_libraries(nvcr_cli PRIVATE CUDA::cudart)
    target_compile_definitions(nvcr_cli PRIVATE NVCR_MEASUREMENT_CUDA=1)
endif()
