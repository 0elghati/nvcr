find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME nvcr_artifact_profiles
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/artifact_tool_tests.py")
    add_test(
        NAME nvcr_engine_catalog
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/engine_catalog_tests.py")
    add_test(
        NAME nvcr_native_installer
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/install_script_tests.py")
    add_test(
        NAME nvcr_onboarding_tools
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/onboarding_tools_tests.py")
    add_test(
        NAME nvcr_softwarex_driver
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/softwarex_driver_tests.py")
else()
    message(WARNING "Python 3 not found; artifact/profile validation test is not registered")
endif()

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

add_executable(nvcr_parser_fuzz_tests parser_fuzz_tests.cpp)
target_link_libraries(nvcr_parser_fuzz_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_parser_fuzz_tests)
add_test(NAME nvcr_parser_fuzz_boundaries COMMAND nvcr_parser_fuzz_tests)

add_executable(
    nvcr_contract_tests
    contract_tests.cpp
    support/test_codec/test_codec.cpp
    support/test_provider/test_provider.cpp)
target_link_libraries(nvcr_contract_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_contract_tests)
if(NVCR_ENABLE_TENSORRT)
    target_compile_definitions(nvcr_contract_tests PRIVATE NVCR_TEST_HAS_TENSORRT=1)
endif()
add_test(NAME nvcr_contract_tests COMMAND nvcr_contract_tests)

add_executable(nvcr_format_contract_tests format_contract_tests.cpp)
target_link_libraries(nvcr_format_contract_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_format_contract_tests)
add_test(NAME nvcr_format_contract_tests COMMAND nvcr_format_contract_tests)

add_executable(
    nvcr_artifact_resolver_tests
    artifact_resolver_tests.cpp
    support/test_provider/test_provider.cpp)
target_link_libraries(nvcr_artifact_resolver_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_artifact_resolver_tests)
add_test(NAME nvcr_artifact_resolver_tests COMMAND nvcr_artifact_resolver_tests)

add_executable(nvcr_artifact_catalog_tests artifact_catalog_tests.cpp)
target_link_libraries(nvcr_artifact_catalog_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_artifact_catalog_tests)
add_test(NAME nvcr_artifact_catalog_tests COMMAND nvcr_artifact_catalog_tests)

if(NVCR_ENABLE_FUZZING)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        add_executable(nvcr_access_unit_fuzz fuzz/access_unit_fuzz.cpp)
        target_link_libraries(nvcr_access_unit_fuzz PRIVATE NVCR::nvcr)
        target_compile_options(nvcr_access_unit_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
        target_link_options(nvcr_access_unit_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
        nvcr_set_project_warnings(nvcr_access_unit_fuzz)
    else()
        message(WARNING "NVCR_ENABLE_FUZZING is ON but libFuzzer target is only wired for Clang")
    endif()
endif()

nvcr_set_project_warnings(nvcr_tests)

add_executable(nvcr_dcvcrt_payload_tests dcvcrt_payload_tests.cpp)
target_include_directories(
    nvcr_dcvcrt_payload_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt")
target_link_libraries(nvcr_dcvcrt_payload_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_dcvcrt_payload_tests)
add_test(NAME nvcr_dcvcrt_payloads COMMAND nvcr_dcvcrt_payload_tests)

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
        nvcr_cuda_ops_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt/backend/tensorrt")
    target_link_libraries(nvcr_cuda_ops_tests PRIVATE NVCR::nvcr CUDA::cudart)
    nvcr_set_project_warnings(nvcr_cuda_ops_tests)
    add_test(NAME nvcr_cuda_ops COMMAND nvcr_cuda_ops_tests)

    add_executable(nvcr_tensorrt_engine_tests tensorrt_engine_tests.cpp)
    target_link_libraries(nvcr_tensorrt_engine_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_tensorrt_engine_tests)

    add_executable(nvcr_tensorrt_roundtrip_tests tensorrt_roundtrip_tests.cpp)
    target_link_libraries(nvcr_tensorrt_roundtrip_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_tensorrt_roundtrip_tests)
    function(nvcr_register_tensorrt_profile_tests engine_dir required)
        set(manifest_path "${engine_dir}/engine_manifest.json")
        if(NOT EXISTS "${manifest_path}")
            if(required)
                message(FATAL_ERROR "TensorRT engine test directory has no manifest: ${engine_dir}")
            endif()
            message(WARNING
                "Skipping unavailable additional TensorRT engine test directory: ${engine_dir}")
            return()
        endif()
        file(READ "${manifest_path}" engine_manifest)
        string(JSON engine_profile_id GET "${engine_manifest}" engine_profile_id)
        string(JSON optimum_width GET "${engine_manifest}" visible_dimensions optimum 0)
        string(JSON optimum_height GET "${engine_manifest}" visible_dimensions optimum 1)
        string(MAKE_C_IDENTIFIER "${engine_profile_id}" profile_test_id)
        string(REGEX REPLACE "^_+" "" profile_test_id "${profile_test_id}")
        set(profile_test_suffix "_${profile_test_id}")
        get_property(registered_profiles GLOBAL PROPERTY NVCR_TENSORRT_TEST_PROFILES)
        if(engine_profile_id IN_LIST registered_profiles)
            return()
        endif()
        set_property(GLOBAL APPEND PROPERTY NVCR_TENSORRT_TEST_PROFILES "${engine_profile_id}")
        add_test(NAME nvcr_tensorrt_engine_contracts${profile_test_suffix}
            COMMAND nvcr_tensorrt_engine_tests "${engine_dir}")
        add_test(NAME nvcr_tensorrt_ip_frame_roundtrip${profile_test_suffix}
            COMMAND nvcr_tensorrt_roundtrip_tests "${engine_dir}"
                "${optimum_width}" "${optimum_height}")
        set_tests_properties(
            nvcr_tensorrt_engine_contracts${profile_test_suffix}
            nvcr_tensorrt_ip_frame_roundtrip${profile_test_suffix}
            PROPERTIES LABELS "gpu;profile-${engine_profile_id}")
    endfunction()

    if(NVCR_TENSORRT_ENGINE_DIR)
        nvcr_register_tensorrt_profile_tests("${NVCR_TENSORRT_ENGINE_DIR}" TRUE)
    endif()
    foreach(engine_dir IN LISTS NVCR_TENSORRT_ENGINE_DIRS)
        if(NOT engine_dir STREQUAL NVCR_TENSORRT_ENGINE_DIR)
            nvcr_register_tensorrt_profile_tests("${engine_dir}" FALSE)
        endif()
    endforeach()
    if(Python3_Interpreter_FOUND
       AND TARGET nvcr_cli
       AND NVCR_DCVCRT_ROOT
       AND NVCR_TENSORRT_ENGINE_DIR
       AND NVCR_DCVCRT_720P_GOLDEN_INPUT)
        add_test(
            NAME nvcr_dcvcrt_i_frame_golden
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/dcvcrt/i_frame_golden.py"
                --nvcr $<TARGET_FILE:nvcr_cli>
                --dcvcrt-root "${NVCR_DCVCRT_ROOT}"
                --python "${NVCR_DCVCRT_ROOT}/src/venv/bin/python3"
                --input "${NVCR_DCVCRT_720P_GOLDEN_INPUT}"
                --engine-dir "${NVCR_TENSORRT_ENGINE_DIR}"
                --output "${CMAKE_CURRENT_BINARY_DIR}/dcvcrt-i-frame-golden.json")
        set_tests_properties(
            nvcr_dcvcrt_i_frame_golden
            PROPERTIES
                LABELS "gpu;conformance"
                TIMEOUT 180)
    endif()
endif()


if(TARGET dcvcrt_reference)
    add_test(NAME dcvcrt_reference_help COMMAND dcvcrt_reference --help)
    if(NVCR_DCVCRT_ROOT)
        add_test(NAME dcvcrt_reference_doctor COMMAND dcvcrt_reference doctor)
    endif()
endif()
