find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME nvcr_dcvcuf_reference_capture
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/dcvcuf_reference_capture_tests.py")
    add_test(
        NAME nvcr_dcvcuf_tensorrt_stages
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/dcvcuf_tensorrt_stage_tests.py")
    add_test(
        NAME nvcr_dcvcuf_architecture_boundary
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/dcvcuf_architecture_tests.py")
    add_test(
        NAME nvcr_artifact_profiles
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/artifact_tool_tests.py")
    add_test(
        NAME nvcr_version_identity
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/version_identity_tests.py")
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
target_compile_definitions(
    nvcr_contract_tests PRIVATE
        NVCR_TEST_PROJECT_VERSION="${PROJECT_VERSION}"
        NVCR_TEST_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        NVCR_TEST_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        NVCR_TEST_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH})
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

add_executable(nvcr_dcvcrt_orchestration_tests dcvcrt_orchestration_tests.cpp)
target_include_directories(
    nvcr_dcvcrt_orchestration_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt")
target_link_libraries(nvcr_dcvcrt_orchestration_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_dcvcrt_orchestration_tests)
add_test(NAME nvcr_dcvcrt_orchestration COMMAND nvcr_dcvcrt_orchestration_tests)

add_executable(nvcr_dcvcrt_host_ops_tests dcvcrt_host_ops_tests.cpp)
target_include_directories(
    nvcr_dcvcrt_host_ops_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt")
target_link_libraries(nvcr_dcvcrt_host_ops_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_dcvcrt_host_ops_tests)
add_test(NAME nvcr_dcvcrt_host_ops COMMAND nvcr_dcvcrt_host_ops_tests)

add_executable(nvcr_mlvc_unit_tests mlvc_unit_tests.cpp)
target_include_directories(nvcr_mlvc_unit_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/mlvc")
target_link_libraries(nvcr_mlvc_unit_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_mlvc_unit_tests)
add_test(NAME nvcr_mlvc_unit COMMAND nvcr_mlvc_unit_tests)

if(NVCR_ENABLE_ONNXRUNTIME)
    add_executable(nvcr_onnxruntime_roundtrip_tests onnxruntime_roundtrip_tests.cpp)
    target_include_directories(
        nvcr_onnxruntime_roundtrip_tests PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt")
    target_link_libraries(nvcr_onnxruntime_roundtrip_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_onnxruntime_roundtrip_tests)
    add_test(NAME nvcr_onnxruntime_session_contract
        COMMAND nvcr_onnxruntime_roundtrip_tests onnxruntime-cpu)
    if(NVCR_ONNXRUNTIME_MODEL_DIR)
        add_test(NAME nvcr_onnxruntime_cpu_ip_frame_roundtrip
            COMMAND nvcr_onnxruntime_roundtrip_tests onnxruntime-cpu
                "${NVCR_ONNXRUNTIME_MODEL_DIR}")
        set_tests_properties(
            nvcr_onnxruntime_cpu_ip_frame_roundtrip
            PROPERTIES LABELS "onnxruntime" TIMEOUT 900)
    endif()

    add_executable(nvcr_mlvc_parity_tests mlvc_parity_tests.cpp)
    target_link_libraries(nvcr_mlvc_parity_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_mlvc_parity_tests)
    # NVCR_MLVC_PARITY_CASES is a list of "BUNDLE|REFERENCE|INPUT|W|H|FRAMES|PERIOD"
    # entries, optionally ending in "|decode-only", recorded with
    # scripts/backends/mlvc/reference_vectors.py.
    set(mlvc_case_index 0)
    foreach(mlvc_case IN LISTS NVCR_MLVC_PARITY_CASES)
        string(REPLACE "|" ";" mlvc_case_fields "${mlvc_case}")
        add_test(NAME nvcr_mlvc_cpu_parity_${mlvc_case_index}
            COMMAND nvcr_mlvc_parity_tests onnxruntime-cpu ${mlvc_case_fields})
        set_tests_properties(
            nvcr_mlvc_cpu_parity_${mlvc_case_index}
            PROPERTIES LABELS "onnxruntime;mlvc" TIMEOUT 900)
        math(EXPR mlvc_case_index "${mlvc_case_index} + 1")
    endforeach()
endif()

add_executable(nvcr_deterministic_executor_tests deterministic_executor_tests.cpp)
target_include_directories(nvcr_deterministic_executor_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(nvcr_deterministic_executor_tests PRIVATE NVCR::nvcr)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nvcr_deterministic_executor_tests PRIVATE -ffp-contract=off)
endif()
if(NVCR_ENABLE_TENSORRT)
    target_compile_definitions(nvcr_deterministic_executor_tests PRIVATE NVCR_TEST_HAS_CUDA=1)
    target_link_libraries(nvcr_deterministic_executor_tests PRIVATE CUDA::cudart)
endif()
nvcr_set_project_warnings(nvcr_deterministic_executor_tests)
# NVCR_DETERMINISTIC_PROGRAM_DIR: programs from
# scripts/backends/dcvcrt/export_normative_programs.sh, for real-stage parity;
# defaults to the ONNX Runtime bundle's normative/ directory when it has one.
if(NOT NVCR_DETERMINISTIC_PROGRAM_DIR AND NVCR_ONNXRUNTIME_MODEL_DIR AND
   IS_DIRECTORY "${NVCR_ONNXRUNTIME_MODEL_DIR}/normative")
    set(NVCR_DETERMINISTIC_PROGRAM_DIR "${NVCR_ONNXRUNTIME_MODEL_DIR}/normative")
endif()
add_test(NAME nvcr_deterministic_executor
    COMMAND nvcr_deterministic_executor_tests ${NVCR_DETERMINISTIC_PROGRAM_DIR})

add_executable(nvcr_fx_reference_tests fx_reference_tests.cpp)
target_include_directories(nvcr_fx_reference_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(nvcr_fx_reference_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_fx_reference_tests)
add_test(NAME nvcr_fx_reference COMMAND nvcr_fx_reference_tests)

add_executable(nvcr_fx_program_tests fx_program_tests.cpp)
target_include_directories(nvcr_fx_program_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(nvcr_fx_program_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_fx_program_tests)
add_test(NAME nvcr_fx_program COMMAND nvcr_fx_program_tests)

add_executable(nvcr_numerics_identity_tests numerics_identity_tests.cpp)
target_link_libraries(nvcr_numerics_identity_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_numerics_identity_tests)
add_test(NAME nvcr_numerics_identity COMMAND nvcr_numerics_identity_tests)

add_executable(nvcr_rans_conformance_tests rans_conformance_tests.cpp)
target_link_libraries(nvcr_rans_conformance_tests PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_rans_conformance_tests)
add_test(NAME nvcr_rans_conformance COMMAND nvcr_rans_conformance_tests)

add_executable(nvcr_dcvcuf_rans_conformance_tests dcvcuf_rans_conformance_tests.cpp)
target_include_directories(
    nvcr_dcvcuf_rans_conformance_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/src/dcvcuf/entropy")
target_link_libraries(nvcr_dcvcuf_rans_conformance_tests PRIVATE NVCR::nvcr)
# libnvcr carries the codec-local UF entropy objects only in DCVC-UF builds.
if(NOT NVCR_ENABLE_DCVC_UF)
    target_sources(
        nvcr_dcvcuf_rans_conformance_tests PRIVATE
            $<TARGET_OBJECTS:nvcr_dcvcuf_entropy>
            $<TARGET_OBJECTS:nvcr_dcvcuf_rans>)
endif()
nvcr_set_project_warnings(nvcr_dcvcuf_rans_conformance_tests)
add_test(NAME nvcr_dcvcuf_rans_conformance COMMAND nvcr_dcvcuf_rans_conformance_tests)

# C13 G1 (encoder-internal reconstruction equals decoder output) and the #194
# partly-filled-group lifecycle checks. Needs a GPU, a DCVC-UF bundle and an
# input sequence; scripts/backends/dcvcuf/run_reconstruction_identity.py runs
# the full matrix. The bounded lifecycle check joins CTest when both cache
# variables name local files.
if(NVCR_ENABLE_TENSORRT OR NVCR_ENABLE_DCVC_UF)
    add_executable(nvcr_dcvcuf_reconstruction_identity dcvcuf_reconstruction_identity.cpp)
    target_include_directories(
        nvcr_dcvcuf_reconstruction_identity PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcuf")
    target_link_libraries(nvcr_dcvcuf_reconstruction_identity PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_dcvcuf_reconstruction_identity)
    # C13 cross-decode: decodes a .nvcr file like the CLI and records every UF
    # entropy call (scripts/backends/dcvcuf/run_cross_decode.py).
    add_executable(nvcr_dcvcuf_entropy_trace dcvcuf_entropy_trace.cpp)
    target_include_directories(
        nvcr_dcvcuf_entropy_trace PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcuf")
    target_link_libraries(nvcr_dcvcuf_entropy_trace PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_dcvcuf_entropy_trace)
    set(NVCR_DCVCUF_TEST_BUNDLE "" CACHE FILEPATH
        "DCVC-UF native .pt bundle or TensorRT stage bundle for the GPU lifecycle check")
    set(NVCR_DCVCUF_TEST_QCIF "" CACHE FILEPATH
        "176x144 YUV420P8 sequence of at least 57 frames for the GPU lifecycle check")
    if(EXISTS "${NVCR_DCVCUF_TEST_BUNDLE}" AND EXISTS "${NVCR_DCVCUF_TEST_QCIF}")
        add_test(
            NAME nvcr_dcvcuf_partial_group_lifecycle
            COMMAND nvcr_dcvcuf_reconstruction_identity partial-group
                "${NVCR_DCVCUF_TEST_BUNDLE}" "${NVCR_DCVCUF_TEST_QCIF}" 176 144 32
                "${CMAKE_CURRENT_BINARY_DIR}/dcvcuf_partial_group_lifecycle.json")
    endif()
endif()

if(TARGET nvcr_cli)
    add_test(
        NAME nvcr_cli_version
        COMMAND "${CMAKE_COMMAND}"
            -DNVCR_CLI=$<TARGET_FILE:nvcr_cli>
            -DEXPECTED_VERSION=${PROJECT_VERSION}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/check_cli_version.cmake")
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME nvcr_artifact_cli_version
            COMMAND "${CMAKE_COMMAND}"
                -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
                -DNVCR_ARTIFACT_CLI=${PROJECT_BINARY_DIR}/generated/bin/nvcr-artifacts
                -DEXPECTED_VERSION=${PROJECT_VERSION}
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_artifact_version.cmake")
    endif()
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
        nvcr_cuda_ops_tests PRIVATE
            "${PROJECT_SOURCE_DIR}/src/dcvcrt/backend/tensorrt"
            "${PROJECT_SOURCE_DIR}/src/dcvcrt")
    target_link_libraries(nvcr_cuda_ops_tests PRIVATE NVCR::nvcr CUDA::cudart)
    nvcr_set_project_warnings(nvcr_cuda_ops_tests)
    add_test(NAME nvcr_cuda_ops COMMAND nvcr_cuda_ops_tests)

    add_executable(nvcr_tensorrt_engine_tests tensorrt_engine_tests.cpp)
    target_link_libraries(nvcr_tensorrt_engine_tests PRIVATE NVCR::nvcr)
    nvcr_set_project_warnings(nvcr_tensorrt_engine_tests)

    add_executable(nvcr_tensorrt_session_tests tensorrt_session_tests.cpp)
    target_include_directories(nvcr_tensorrt_session_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(
        nvcr_tensorrt_session_tests PRIVATE NVCR::nvcr CUDA::cudart TensorRT::TensorRT)
    nvcr_set_project_warnings(nvcr_tensorrt_session_tests)

    # Offline per-stage conformance runner (used by C11 DCVC-UF stage checks).
    add_executable(nvcr_tensorrt_stage_runner tensorrt_stage_runner.cpp)
    target_include_directories(nvcr_tensorrt_stage_runner PRIVATE "${PROJECT_SOURCE_DIR}/src")
    target_link_libraries(
        nvcr_tensorrt_stage_runner PRIVATE NVCR::nvcr CUDA::cudart TensorRT::TensorRT)
    nvcr_set_project_warnings(nvcr_tensorrt_stage_runner)

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
        add_test(NAME nvcr_tensorrt_session_contracts${profile_test_suffix}
            COMMAND nvcr_tensorrt_session_tests "${engine_dir}")
        add_test(NAME nvcr_tensorrt_ip_frame_roundtrip${profile_test_suffix}
            COMMAND nvcr_tensorrt_roundtrip_tests "${engine_dir}"
                "${optimum_width}" "${optimum_height}")
        set_tests_properties(
            nvcr_tensorrt_engine_contracts${profile_test_suffix}
            nvcr_tensorrt_session_contracts${profile_test_suffix}
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

    # Issue #216: a stream decoded by a target that computes different entropy
    # parameters must fail cleanly, never reconstruct garbage. TensorRT and
    # ONNX Runtime stand in for two GPUs with different plans. Use the smallest
    # available exact profile so the ONNX Runtime CPU side stays fast.
    if(NVCR_ENABLE_ONNXRUNTIME
       AND NVCR_ONNXRUNTIME_MODEL_DIR
       AND Python3_Interpreter_FOUND
       AND TARGET nvcr_cli)
        set(cross_engine_dir "")
        set(cross_engine_area 0)
        foreach(engine_dir IN LISTS NVCR_TENSORRT_ENGINE_DIR NVCR_TENSORRT_ENGINE_DIRS)
            if(EXISTS "${engine_dir}/engine_manifest.json")
                file(READ "${engine_dir}/engine_manifest.json" engine_manifest)
                string(JSON engine_width GET "${engine_manifest}" visible_dimensions optimum 0)
                string(JSON engine_height GET "${engine_manifest}" visible_dimensions optimum 1)
                math(EXPR engine_area "${engine_width} * ${engine_height}")
                if(cross_engine_area EQUAL 0 OR engine_area LESS cross_engine_area)
                    set(cross_engine_dir "${engine_dir}")
                    set(cross_engine_area ${engine_area})
                    set(cross_engine_width ${engine_width})
                    set(cross_engine_height ${engine_height})
                endif()
            endif()
        endforeach()
        if(cross_engine_dir)
            add_test(
                NAME nvcr_dcvcrt_cross_provider_decode
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/dcvcrt/cross_provider_decode.py"
                    --nvcr $<TARGET_FILE:nvcr_cli>
                    --sample-generator "${PROJECT_SOURCE_DIR}/scripts/generate_sample_yuv.py"
                    --tensorrt-engine-dir "${cross_engine_dir}"
                    --onnxruntime-model-dir "${NVCR_ONNXRUNTIME_MODEL_DIR}"
                    --width ${cross_engine_width}
                    --height ${cross_engine_height}
                    --work-dir "${CMAKE_CURRENT_BINARY_DIR}/cross-provider-decode")
            set_tests_properties(
                nvcr_dcvcrt_cross_provider_decode
                PROPERTIES
                    LABELS "gpu;onnxruntime;conformance"
                    TIMEOUT 1200)
        endif()
    endif()
endif()


if(TARGET dcvcrt_reference)
    add_test(NAME dcvcrt_reference_help COMMAND dcvcrt_reference --help)
    if(NVCR_DCVCRT_ROOT)
        add_test(NAME dcvcrt_reference_doctor COMMAND dcvcrt_reference doctor)
    endif()
endif()

# Cross-check the offline byte accountant against the production writers.
add_executable(nvcr_measurement_wire_fixture measurement_wire_fixture.cpp)
target_include_directories(nvcr_measurement_wire_fixture PRIVATE "${PROJECT_SOURCE_DIR}/src/dcvcrt")
target_link_libraries(nvcr_measurement_wire_fixture PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_measurement_wire_fixture)
if(Python3_Interpreter_FOUND)
    add_test(NAME nvcr_measurement_semantics
        COMMAND "${CMAKE_COMMAND}" -E env
            "NVCR_WIRE_FIXTURE=$<TARGET_FILE:nvcr_measurement_wire_fixture>"
            "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/measurement_tests.py")
    add_test(NAME nvcr_measurement_laptop
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/measurement_laptop_tests.py")
    add_test(NAME nvcr_portability_consolidation
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/portability_results_tests.py")
    add_test(NAME nvcr_legacy_consolidation
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/consolidate_runtime_results_tests.py")
endif()

add_executable(nvcr_public_api_compatibility public_api_compatibility.cpp)
target_link_libraries(nvcr_public_api_compatibility PRIVATE NVCR::nvcr)
nvcr_set_project_warnings(nvcr_public_api_compatibility)
add_test(NAME nvcr_public_api_compatibility COMMAND nvcr_public_api_compatibility)
