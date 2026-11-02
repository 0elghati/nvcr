add_library(nvcr_msrtc_rans OBJECT src/EntropyCoder.cpp)
target_include_directories(
    nvcr_msrtc_rans
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/private/include")
target_compile_features(nvcr_msrtc_rans PUBLIC cxx_std_17)
set_target_properties(
    nvcr_msrtc_rans PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
