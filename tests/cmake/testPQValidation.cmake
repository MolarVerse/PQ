cmake_policy(SET CMP0054 NEW)

function(run_pq_in working_directory output_var error_var result_var)
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E env
            "GMON_OUT_PREFIX=${VALIDATION_GMON_PREFIX}"
            "${PQ_EXECUTABLE}" ${ARGN}
        WORKING_DIRECTORY "${working_directory}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    set(${output_var} "${output}" PARENT_SCOPE)
    set(${error_var} "${error}" PARENT_SCOPE)
    set(${result_var} "${result}" PARENT_SCOPE)
endfunction()

function(snapshot_directory directory output_var)
    file(GLOB_RECURSE relative_paths RELATIVE "${directory}" "${directory}/*")
    list(SORT relative_paths)

    set(snapshot "")
    foreach(relative_path IN LISTS relative_paths)
        set(path "${directory}/${relative_path}")
        if(NOT IS_DIRECTORY "${path}")
            file(SHA256 "${path}" sha256)
            string(APPEND snapshot "${relative_path}:${sha256}\n")
        endif()
    endforeach()

    set(${output_var} "${snapshot}" PARENT_SCOPE)
endfunction()

set(validation_gmon_directory "${VALIDATION_WORK_DIR}-gmon")
set(VALIDATION_GMON_PREFIX "${validation_gmon_directory}/gmon")
file(REMOVE_RECURSE "${validation_gmon_directory}")
file(MAKE_DIRECTORY "${validation_gmon_directory}")

file(REMOVE_RECURSE "${VALIDATION_WORK_DIR}")
file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}")
file(
    COPY
    "${VALIDATION_FIXTURE_DIR}/run.in"
    "${VALIDATION_FIXTURE_DIR}/start.rst"
    "${VALIDATION_FIXTURE_DIR}/moldescriptor.dat"
    "${VALIDATION_FIXTURE_DIR}/guff.dat"
    DESTINATION "${VALIDATION_WORK_DIR}"
)

snapshot_directory("${VALIDATION_WORK_DIR}" before_validation)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate run.in --format=json
)
snapshot_directory("${VALIDATION_WORK_DIR}" after_validation)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ rejected a valid input: ${output} ${error}")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "JSON validation wrote to stderr: ${error}")
endif()
if(NOT before_validation STREQUAL after_validation)
    message(FATAL_ERROR "Input validation changed files")
endif()

string(JSON schema GET "${output}" schema)
string(JSON schema_version GET "${output}" schema_version)
string(JSON valid GET "${output}" valid)
string(JSON scope GET "${output}" scope)
string(JSON diagnostic_count LENGTH "${output}" diagnostics)
if(NOT schema STREQUAL "pq.validation" OR NOT schema_version EQUAL 1)
    message(FATAL_ERROR "Unexpected validation schema: ${output}")
endif()
if(NOT valid OR NOT scope STREQUAL "installed")
    message(FATAL_ERROR "Unexpected valid input result: ${output}")
endif()
if(NOT diagnostic_count EQUAL 0)
    message(FATAL_ERROR "Valid input produced diagnostics: ${output}")
endif()

run_pq_in("${VALIDATION_WORK_DIR}" output error result --validate run.in)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Text validation rejected a valid input: ${error}")
endif()
if(NOT output STREQUAL "Valid PQ input: run.in\n")
    message(FATAL_ERROR "Unexpected text validation output: ${output}")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "Valid text validation wrote to stderr: ${error}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/invalid.in"
    "jobtype = mm-md;\n"
    "not_a_keyword = 1;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate invalid.in --format=json
)
if(NOT result EQUAL 1)
    message(FATAL_ERROR "Invalid input returned ${result}: ${output} ${error}")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "Invalid JSON validation wrote to stderr: ${error}")
endif()
string(JSON valid GET "${output}" valid)
string(JSON severity GET "${output}" diagnostics 0 severity)
string(JSON file GET "${output}" diagnostics 0 file)
string(JSON line GET "${output}" diagnostics 0 line)
if(valid OR NOT severity STREQUAL "error")
    message(FATAL_ERROR "Invalid input produced the wrong result: ${output}")
endif()
if(NOT file STREQUAL "invalid.in" OR NOT line EQUAL 2)
    message(FATAL_ERROR "Invalid input lost its source location: ${output}")
endif()

file(READ "${VALIDATION_FIXTURE_DIR}/run.in" portable_input)
string(
    REPLACE "start_file = start.rst;" "start_file = missing.rst;"
    portable_input "${portable_input}"
)
file(WRITE "${VALIDATION_WORK_DIR}/portable.in" "${portable_input}")

run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate portable.in --format=json --scope=portable
)
if(NOT result EQUAL 0 OR NOT error STREQUAL "")
    message(FATAL_ERROR "Portable validation required a local file: ${output} ${error}")
endif()
string(JSON scope GET "${output}" scope)
if(NOT scope STREQUAL "portable")
    message(FATAL_ERROR "Portable validation reported the wrong scope: ${output}")
endif()

run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate portable.in --format=json --scope=installed
)
if(NOT result EQUAL 1 OR NOT error STREQUAL "")
    message(FATAL_ERROR "Installed validation ignored a missing file: ${output} ${error}")
endif()
string(JSON message GET "${output}" diagnostics 0 message)
if(NOT message MATCHES "missing.rst")
    message(FATAL_ERROR "Missing file diagnostic is unclear: ${output}")
endif()
