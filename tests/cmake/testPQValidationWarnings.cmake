function(assert_warning input_file scope expected_message)
    set(arguments --validate "${input_file}" --format=json)
    if(scope STREQUAL "portable")
        list(APPEND arguments --scope=portable)
    endif()

    execute_process(
        COMMAND "${PQ_EXECUTABLE}" ${arguments}
        WORKING_DIRECTORY "${VALIDATION_WORK_DIR}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )

    if(NOT result EQUAL 0 OR NOT error STREQUAL "")
        message(
            FATAL_ERROR
            "Validation rejected ${input_file}: ${output} ${error}"
        )
    endif()

    string(JSON valid GET "${output}" valid)
    string(JSON diagnostic_count LENGTH "${output}" diagnostics)
    string(JSON severity GET "${output}" diagnostics 0 severity)
    string(JSON message GET "${output}" diagnostics 0 message)

    if(
        NOT valid
        OR NOT diagnostic_count EQUAL 1
        OR NOT severity STREQUAL "warning"
        OR NOT message MATCHES "${expected_message}"
    )
        message(FATAL_ERROR "Unexpected warning for ${input_file}: ${output}")
    endif()
endfunction()

file(REMOVE_RECURSE "${VALIDATION_WORK_DIR}")
file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}")
file(
    COPY
    "${VALIDATION_FIXTURE_DIR}/start.rst"
    "${VALIDATION_FIXTURE_DIR}/moldescriptor.dat"
    "${VALIDATION_FIXTURE_DIR}/guff.dat"
    DESTINATION "${VALIDATION_WORK_DIR}"
)

file(
    WRITE "${VALIDATION_WORK_DIR}/nose-hoover.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = off;\n"
    "thermostat = nh-chain;\n"
    "temp = 300;\n"
    "coupling_frequency = 0;\n"
    "start_file = start.rst;\n"
)
assert_warning(
    "nose-hoover.in"
    portable
    "zero Nose-Hoover coupling frequency"
)

file(
    WRITE "${VALIDATION_WORK_DIR}/langevin.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = off;\n"
    "thermostat = langevin;\n"
    "temp = 300;\n"
    "friction = 0;\n"
    "start_file = start.rst;\n"
)
assert_warning("langevin.in" installed "zero Langevin friction")

file(
    WRITE "${VALIDATION_WORK_DIR}/manostat.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = off;\n"
    "manostat = berendsen;\n"
    "pressure = 1;\n"
    "compressibility = 0;\n"
    "start_file = start.rst;\n"
)
assert_warning("manostat.in" installed "zero compressibility")
