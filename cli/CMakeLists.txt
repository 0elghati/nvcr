add_executable(nvcr_cli main.cpp)
set_target_properties(nvcr_cli PROPERTIES OUTPUT_NAME nvcr)
target_link_libraries(nvcr_cli PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_cli)
