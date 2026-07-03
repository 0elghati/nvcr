if(NOT DEFINED NVCR_CLI OR NOT DEFINED TEST_DIRECTORY)
    message(FATAL_ERROR "NVCR_CLI and TEST_DIRECTORY are required")
endif()

set(output "${TEST_DIRECTORY}/implicit-all-intra.nvcr")
file(REMOVE "${output}")
execute_process(
    COMMAND "${NVCR_CLI}" encode
        -i "${TEST_DIRECTORY}/missing-input.yuv"
        -o "${output}"
        -s 176x144 --frames 2 --engine-dir "${TEST_DIRECTORY}/missing-engines"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)

if(result EQUAL 0)
    message(FATAL_ERROR "missing input unexpectedly encoded")
endif()
set(diagnostic "${stdout}${stderr}")
if(diagnostic MATCHES "predicted-frame encoding is not implemented" OR
   diagnostic MATCHES "refusing to silently encode")
    message(FATAL_ERROR "inter GOP is still rejected: ${diagnostic}")
endif()
if(NOT diagnostic MATCHES "cannot open input")
    message(FATAL_ERROR "multi-frame request did not pass GOP validation: ${diagnostic}")
endif()
if(EXISTS "${output}")
    message(FATAL_ERROR "rejected encoding created an output file")
endif()
