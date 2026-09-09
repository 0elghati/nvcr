if(NOT DEFINED NVCR_CLI OR NOT DEFINED EXPECTED_VERSION)
    message(FATAL_ERROR "NVCR_CLI and EXPECTED_VERSION are required")
endif()

execute_process(
    COMMAND "${NVCR_CLI}" --version
    RESULT_VARIABLE command_result
    OUTPUT_VARIABLE command_output
    ERROR_VARIABLE command_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
if(NOT command_result EQUAL 0)
    message(FATAL_ERROR "nvcr --version exited with ${command_result}")
endif()
if(NOT command_error STREQUAL "")
    message(FATAL_ERROR "nvcr --version wrote stderr: ${command_error}")
endif()
set(expected_output "nvcr ${EXPECTED_VERSION}")
if(NOT command_output STREQUAL expected_output)
    message(FATAL_ERROR
        "nvcr --version expected '${expected_output}', got "
        "'${command_output}'")
endif()
