if(TARGET GTest::gtest_main)
    add_executable(nvcr_tests unit_tests.cpp)
    target_link_libraries(nvcr_tests PRIVATE NVCR::nvcr GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(nvcr_tests)
else()
    message(STATUS "GoogleTest not found; building dependency-free NVCR smoke tests")
    add_executable(nvcr_tests smoke_tests.cpp)
    target_link_libraries(nvcr_tests PRIVATE NVCR::nvcr)
    add_test(NAME nvcr_smoke_tests COMMAND nvcr_tests)
endif()
nvcr_set_project_warnings(nvcr_tests)

add_executable(nvcr_rans_conformance_tests rans_conformance_tests.cpp)
target_link_libraries(nvcr_rans_conformance_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_rans_conformance_tests)
add_test(NAME nvcr_rans_conformance COMMAND nvcr_rans_conformance_tests)

if(TARGET nvcr_cli)
    add_test(
        NAME nvcr_cli_accepts_inter_gop
        COMMAND "${CMAKE_COMMAND}"
            -DNVCR_CLI=$<TARGET_FILE:nvcr_cli>
            -DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_cli_gop.cmake")
endif()

if(NVCR_ENABLE_TENSORRT)
    add_executable(nvcr_cuda_ops_tests cuda_ops_tests.cpp)
    target_include_directories(
        nvcr_cuda_ops_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt")
    target_link_libraries(nvcr_cuda_ops_tests PRIVATE NVCR::nvcr CUDA::cudart)
    nvcr_set_project_warnings(nvcr_cuda_ops_tests)
    add_test(NAME nvcr_cuda_ops COMMAND nvcr_cuda_ops_tests)

    add_executable(nvcr_tensorrt_engine_tests tensorrt_engine_tests.cpp)
    target_link_libraries(nvcr_tensorrt_engine_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_tensorrt_engine_tests)

    add_executable(nvcr_tensorrt_roundtrip_tests tensorrt_roundtrip_tests.cpp)
    target_link_libraries(nvcr_tensorrt_roundtrip_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_tensorrt_roundtrip_tests)
    if(NVCR_TENSORRT_ENGINE_DIR)
        add_test(NAME nvcr_tensorrt_engine_contracts
            COMMAND nvcr_tensorrt_engine_tests "${NVCR_TENSORRT_ENGINE_DIR}")
        add_test(NAME nvcr_tensorrt_ip_frame_roundtrip
            COMMAND nvcr_tensorrt_roundtrip_tests "${NVCR_TENSORRT_ENGINE_DIR}")
    endif()
endif()


if(TARGET dcvcrt_reference)
    add_test(NAME dcvcrt_reference_help COMMAND dcvcrt_reference --help)
    if(NVCR_DCVCRT_ROOT)
        add_test(NAME dcvcrt_reference_doctor COMMAND dcvcrt_reference doctor)
    endif()
endif()
