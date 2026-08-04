function(run_validation input_file scope output_var result_var)
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

    if(NOT error STREQUAL "")
        message(FATAL_ERROR "Validation wrote to stderr: ${error}")
    endif()

    set(${output_var} "${output}" PARENT_SCOPE)
    set(${result_var} "${result}" PARENT_SCOPE)
endfunction()

function(assert_valid input_file scope)
    run_validation("${input_file}" "${scope}" output result)
    if(NOT result EQUAL 0)
        message(
            FATAL_ERROR
            "${scope} validation rejected ${input_file}: ${output}"
        )
    endif()

    string(JSON valid GET "${output}" valid)
    if(NOT valid)
        message(FATAL_ERROR "${input_file} was not reported valid: ${output}")
    endif()
endfunction()

function(assert_invalid input_file scope expected_message)
    run_validation("${input_file}" "${scope}" output result)
    if(NOT result EQUAL 1)
        message(
            FATAL_ERROR
            "${scope} validation accepted ${input_file}: ${output}"
        )
    endif()

    string(JSON message GET "${output}" diagnostics 0 message)
    if(NOT message MATCHES "${expected_message}")
        message(
            FATAL_ERROR
            "Unexpected diagnostic for ${input_file}: ${output}"
        )
    endif()
endfunction()

file(REMOVE_RECURSE "${VALIDATION_WORK_DIR}")
file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}")
file(
    COPY "${VALIDATION_FIXTURE_DIR}/start.rst"
    DESTINATION "${VALIDATION_WORK_DIR}"
)

file(
    WRITE "${VALIDATION_WORK_DIR}/mm.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = off;\n"
    "start_file = start.rst;\n"
)
assert_invalid("mm.in" installed "Moldescriptor file.*does not exist")
file(WRITE "${VALIDATION_WORK_DIR}/moldescriptor.dat" "")
assert_invalid("mm.in" installed "Guff file.*does not exist")
file(WRITE "${VALIDATION_WORK_DIR}/guff.dat" "")
assert_valid("mm.in" installed)

file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}/start-directory")
file(
    WRITE "${VALIDATION_WORK_DIR}/directory-reference.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = off;\n"
    "start_file = start-directory;\n"
)
assert_invalid(
    "directory-reference.in"
    installed
    "Cannot open start file.*start-directory"
)

file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}/topology-directory")
file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}/parameter-directory")
file(
    WRITE "${VALIDATION_WORK_DIR}/force-field.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = on;\n"
    "topology_file = topology-directory;\n"
    "parameter_file = parameter-directory;\n"
    "start_file = start.rst;\n"
)
assert_invalid(
    "force-field.in"
    installed
    "Cannot open topology file.*topology-directory"
)
file(REMOVE_RECURSE "${VALIDATION_WORK_DIR}/topology-directory")
file(WRITE "${VALIDATION_WORK_DIR}/topology-directory" "")
assert_invalid(
    "force-field.in"
    installed
    "Cannot open parameter file.*parameter-directory"
)
file(REMOVE_RECURSE "${VALIDATION_WORK_DIR}/parameter-directory")
file(WRITE "${VALIDATION_WORK_DIR}/parameter-directory" "")
assert_valid("force-field.in" installed)

file(
    WRITE "${VALIDATION_WORK_DIR}/dftb.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "qm_script = dftbplus_periodic_stress;\n"
    "start_file = start.rst;\n"
)
assert_valid("dftb.in" portable)
assert_invalid("dftb.in" installed "DFTB setup file.*does not exist")
file(WRITE "${VALIDATION_WORK_DIR}/dftb_in.template" "")

if(EXPECTED_SHARED AND NOT EXPECTED_SINGULARITY)
    assert_valid("dftb.in" installed)

    set(relocated_prefix "${VALIDATION_WORK_DIR}/relocated")
    file(MAKE_DIRECTORY "${relocated_prefix}/bin")
    file(MAKE_DIRECTORY "${relocated_prefix}/share/PQ/scripts")
    file(
        COPY "${PQ_EXECUTABLE}"
        DESTINATION "${relocated_prefix}/bin"
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    )
    file(
        COPY "${BUNDLED_SCRIPT_SOURCE_DIR}/dftbplus_periodic_stress"
        DESTINATION "${relocated_prefix}/share/PQ/scripts"
    )

    get_filename_component(pq_executable_name "${PQ_EXECUTABLE}" NAME)
    set(build_tree_pq "${PQ_EXECUTABLE}")
    set(PQ_EXECUTABLE "${relocated_prefix}/bin/${pq_executable_name}")
    assert_valid("dftb.in" installed)
    file(
        REMOVE
        "${relocated_prefix}/share/PQ/scripts/dftbplus_periodic_stress"
    )
    assert_invalid(
        "dftb.in"
        installed
        "Bundled QM script.*does not exist"
    )
    set(PQ_EXECUTABLE "${build_tree_pq}")
else()
    assert_invalid(
        "dftb.in"
        installed
        "requires.*qm_script_full_path"
    )
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/wrong-script.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = pyscf;\n"
    "qm_script = dftbplus_periodic_stress;\n"
    "start_file = start.rst;\n"
)
assert_invalid(
    "wrong-script.in"
    portable
    "not available for pyscf"
)

file(
    WRITE "${VALIDATION_WORK_DIR}/two-scripts.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "qm_script = dftbplus_periodic_stress;\n"
    "qm_script_full_path = runner;\n"
    "start_file = start.rst;\n"
)
assert_invalid("two-scripts.in" portable "mutually exclusive")

file(
    WRITE "${VALIDATION_WORK_DIR}/missing-script.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "start_file = start.rst;\n"
)
assert_invalid("missing-script.in" portable "No qm_script provided")

file(
    WRITE "${VALIDATION_WORK_DIR}/full-path.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "qm_script_full_path = runner;\n"
    "dftb_file = dftb_in.template;\n"
    "start_file = start.rst;\n"
)
assert_valid("full-path.in" portable)
assert_invalid("full-path.in" installed "QM script.*does not exist")
file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}/runner")
assert_invalid("full-path.in" installed "QM script.*not a regular file")
file(REMOVE_RECURSE "${VALIDATION_WORK_DIR}/runner")
file(WRITE "${VALIDATION_WORK_DIR}/runner" "")
assert_valid("full-path.in" installed)

file(
    WRITE "${VALIDATION_WORK_DIR}/turbomole.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = turbomole;\n"
    "qm_script = turbomole_rimp2;\n"
    "start_file = start.rst;\n"
)
assert_valid("turbomole.in" portable)
if(EXPECTED_SHARED AND NOT EXPECTED_SINGULARITY)
    assert_invalid(
        "turbomole.in"
        installed
        "Required QM working file.*tm_define.template"
    )
    file(WRITE "${VALIDATION_WORK_DIR}/tm_define.template" "")
    assert_valid("turbomole.in" installed)
endif()

foreach(slakos IN ITEMS 3ob matsci)
    file(
        WRITE "${VALIDATION_WORK_DIR}/${slakos}.in"
        "jobtype = qm-md;\n"
        "nstep = 1;\n"
        "timestep = 0.5;\n"
        "qm_prog = ase-dftbplus;\n"
        "slakos = ${slakos};\n"
        "start_file = start.rst;\n"
    )
    assert_valid("${slakos}.in" portable)

    if(EXPECTED_ASE)
        assert_valid("${slakos}.in" installed)
    else()
        assert_invalid("${slakos}.in" installed "requires ASE support")
    endif()
endforeach()

file(
    WRITE "${VALIDATION_WORK_DIR}/custom-slakos.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = ase-dftbplus;\n"
    "slakos = custom;\n"
    "slakos_path = custom-sk;\n"
    "start_file = start.rst;\n"
)
assert_valid("custom-slakos.in" portable)
if(EXPECTED_ASE)
    assert_invalid(
        "custom-slakos.in"
        installed
        "Slater-Koster directory.*does not exist"
    )
else()
    assert_invalid("custom-slakos.in" installed "requires ASE support")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/fennol.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = fennol;\n"
    "fennol_model_path = missing.fnx;\n"
    "start_file = start.rst;\n"
)
assert_valid("fennol.in" portable)
if(EXPECTED_ASE)
    assert_invalid("fennol.in" installed "FeNNol model file.*does not exist")
else()
    assert_invalid("fennol.in" installed "requires ASE support")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/mace-local.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = mace;\n"
    "mace_model = custom;\n"
    "mace_model_path = missing.model;\n"
    "start_file = start.rst;\n"
)
assert_valid("mace-local.in" portable)
if(EXPECTED_ASE)
    assert_invalid(
        "mace-local.in"
        installed
        "MACE model file.*does not exist"
    )
else()
    assert_invalid("mace-local.in" installed "requires ASE support")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/mace-remote.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = mace;\n"
    "mace_model = custom;\n"
    "mace_model_path = https://example.org/model.model;\n"
    "start_file = start.rst;\n"
)
assert_valid("mace-remote.in" portable)
if(EXPECTED_ASE)
    assert_valid("mace-remote.in" installed)
else()
    assert_invalid("mace-remote.in" installed "requires ASE support")
endif()
