find_package(Threads REQUIRED)

add_library(nvcr_dcvc_rans OBJECT src/rans.cpp)
target_include_directories(nvcr_dcvc_rans PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(nvcr_dcvc_rans PUBLIC cxx_std_17)
target_link_libraries(nvcr_dcvc_rans PUBLIC Threads::Threads)
set_target_properties(
    nvcr_dcvc_rans PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)

