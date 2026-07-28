function(run_pq_in working_directory output_var error_var result_var)
    execute_process(
        COMMAND "${PQ_EXECUTABLE}" ${ARGN}
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
string(JSON validation_schema GET "${output}" schema)
string(JSON validation_valid GET "${output}" valid)
string(JSON validation_scope GET "${output}" scope)
string(JSON diagnostic_count LENGTH "${output}" diagnostics)
if(NOT validation_schema STREQUAL "pq.validation")
    message(FATAL_ERROR "Unexpected validation schema: ${validation_schema}")
endif()
if(NOT validation_valid)
    message(FATAL_ERROR "Valid input reported invalid: ${output}")
endif()
if(NOT validation_scope STREQUAL "installed")
    message(FATAL_ERROR "Unexpected default validation scope: ${output}")
endif()
if(NOT diagnostic_count EQUAL 0)
    message(FATAL_ERROR "Valid input produced diagnostics: ${output}")
endif()

run_pq_in(
    "${VALIDATION_WORK_DIR}"
    capabilities_output capabilities_error capabilities_result
    --capabilities=json
)
if(NOT capabilities_result EQUAL 0 OR NOT capabilities_error STREQUAL "")
    message(
        FATAL_ERROR
        "Could not read PQ capabilities: ${capabilities_error}"
    )
endif()
string(
    JSON t_relaxation_max
    GET "${capabilities_output}" input parameters t_relaxation maximum
)
string(
    JSON friction_max
    GET "${capabilities_output}" input parameters friction maximum
)
string(
    JSON coupling_frequency_max
    GET "${capabilities_output}" input parameters coupling_frequency maximum
)
string(
    JSON p_relaxation_max
    GET "${capabilities_output}" input parameters p_relaxation maximum
)
string(
    JSON dftbplus_recommended_script
    GET "${capabilities_output}" input external_qm programs dftbplus recommended_script
)
string(
    JSON pyscf_script_count
    LENGTH "${capabilities_output}" input external_qm programs pyscf scripts
)
string(
    JSON pyscf_recommended_type
    TYPE "${capabilities_output}" input external_qm programs pyscf recommended_script
)
if(NOT dftbplus_recommended_script STREQUAL "dftbplus_periodic_stress")
    message(FATAL_ERROR "Unexpected DFTB+ recommendation: ${capabilities_output}")
endif()
if(NOT pyscf_script_count EQUAL 2)
    message(FATAL_ERROR "Unexpected PySCF script list: ${capabilities_output}")
endif()
if(NOT pyscf_recommended_type STREQUAL "NULL")
    message(FATAL_ERROR "PySCF must not select a method implicitly: ${capabilities_output}")
endif()
file(READ "${VALIDATION_FIXTURE_DIR}/run.in" maximum_input)
string(
    APPEND maximum_input
    "\nt_relaxation = ${t_relaxation_max};\n"
    "friction = ${friction_max};\n"
    "coupling_frequency = ${coupling_frequency_max};\n"
    "p_relaxation = ${p_relaxation_max};\n"
)
file(
    WRITE "${VALIDATION_WORK_DIR}/advertised-maxima.in"
    "${maximum_input}"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate advertised-maxima.in --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ rejected an advertised maximum: ${output}")
endif()

file(READ "${VALIDATION_FIXTURE_DIR}/run.in" unsafe_thermostat_input)
string(
    APPEND unsafe_thermostat_input
    "\nthermostat = berendsen;\n"
    "temp = 300;\n"
    "t_relaxation = 0.0001;\n"
)
file(
    WRITE "${VALIDATION_WORK_DIR}/unsafe-thermostat.in"
    "${unsafe_thermostat_input}"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate unsafe-thermostat.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a timestep above the thermostat relaxation time")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "timestep must not exceed")
    message(FATAL_ERROR "Unexpected thermostat stability diagnostic: ${output}")
endif()

file(READ "${VALIDATION_FIXTURE_DIR}/run.in" unsafe_langevin_input)
string(
    APPEND unsafe_langevin_input
    "\nthermostat = langevin;\n"
    "temp = 300;\n"
    "friction = 1e290;\n"
)
file(
    WRITE "${VALIDATION_WORK_DIR}/unsafe-langevin.in"
    "${unsafe_langevin_input}"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate unsafe-langevin.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a non-finite Langevin scale")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "non-finite random-force scale")
    message(FATAL_ERROR "Unexpected Langevin diagnostic: ${output}")
endif()

file(READ "${VALIDATION_FIXTURE_DIR}/run.in" unsafe_manostat_input)
string(
    APPEND unsafe_manostat_input
    "\nmanostat = berendsen;\n"
    "pressure = 1;\n"
    "p_relaxation = 0.0001;\n"
)
file(
    WRITE "${VALIDATION_WORK_DIR}/unsafe-manostat.in"
    "${unsafe_manostat_input}"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate unsafe-manostat.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a timestep above the manostat relaxation time")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "timestep must not exceed")
    message(FATAL_ERROR "Unexpected manostat stability diagnostic: ${output}")
endif()

file(READ "${VALIDATION_FIXTURE_DIR}/run.in" missing_initial_temp_input)
string(APPEND missing_initial_temp_input "\ninit_velocities = force;\n")
file(
    WRITE "${VALIDATION_WORK_DIR}/missing-initial-temp.in"
    "${missing_initial_temp_input}"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate missing-initial-temp.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted velocity initialization without a temperature")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "Initializing velocities requires")
    message(FATAL_ERROR "Unexpected initialization diagnostic: ${output}")
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
    READ "${VALIDATION_FIXTURE_DIR}/run.in"
    mm_with_unused_qm
)
string(
    APPEND mm_with_unused_qm
    "\nqm_prog = dftbplus;\n"
    "mace_model_size = small;\n"
)
file(
    WRITE
    "${VALIDATION_WORK_DIR}/mm-with-unused-qm.in"
    "${mm_with_unused_qm}"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate mm-with-unused-qm.in --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ rejected unused QM settings in an MM input: ${output}")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "Warning validation wrote to stderr: ${error}")
endif()
string(JSON validation_valid GET "${output}" valid)
string(JSON diagnostic_count LENGTH "${output}" diagnostics)
string(JSON diagnostic_severity GET "${output}" diagnostics 0 severity)
if(NOT validation_valid OR NOT diagnostic_count EQUAL 1)
    message(FATAL_ERROR "Unexpected warning validation result: ${output}")
endif()
if(NOT diagnostic_severity STREQUAL "warning")
    message(FATAL_ERROR "Unexpected diagnostic severity: ${output}")
endif()

set(qm_without_descriptor_dir "${VALIDATION_WORK_DIR}/qm-no-descriptor")
file(MAKE_DIRECTORY "${qm_without_descriptor_dir}")
file(
    COPY
    "${VALIDATION_FIXTURE_DIR}/start.rst"
    DESTINATION "${qm_without_descriptor_dir}"
)
if(EXPECTED_SHARED AND NOT EXPECTED_SINGULARITY)
    set(qm_script_setting "qm_script = dftbplus_periodic_stress;")
else()
    file(WRITE "${qm_without_descriptor_dir}/qm-runner" "")
    set(
        qm_script_setting
        "qm_script_full_path = ${qm_without_descriptor_dir}/qm-runner;"
    )
endif()
file(
    WRITE "${qm_without_descriptor_dir}/run.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "${qm_script_setting}\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate run.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted direct DFTB+ without a setup file")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "DFTB setup file.*does not exist")
    message(FATAL_ERROR "Unexpected DFTB setup diagnostic: ${output}")
endif()
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate run.in --format=json --scope=portable
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation required a local DFTB setup: ${output}")
endif()
string(JSON validation_scope GET "${output}" scope)
if(NOT validation_scope STREQUAL "portable")
    message(FATAL_ERROR "Portable validation reported the wrong scope: ${output}")
endif()
file(WRITE "${qm_without_descriptor_dir}/dftb_in.template" "")
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate run.in --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ required a moldescriptor for pure QM: ${output}")
endif()

file(
    WRITE "${qm_without_descriptor_dir}/custom-slakos.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = ase-dftbplus;\n"
    "slakos = custom;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate custom-slakos.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted custom Slater-Koster parameters without a path")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "require.*slakos_path")
    message(FATAL_ERROR "Unexpected custom Slater-Koster diagnostic: ${output}")
endif()

file(
    WRITE "${qm_without_descriptor_dir}/missing-slakos.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = ase-dftbplus;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate missing-slakos.in --scope=portable --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted ASE-DFTB+ without a Slater-Koster set")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "requires slakos")
    message(FATAL_ERROR "Unexpected missing Slater-Koster diagnostic: ${output}")
endif()

if(EXPECTED_ASE)
    file(
        WRITE "${qm_without_descriptor_dir}/custom-slakos.in"
        "jobtype = qm-md;\n"
        "nstep = 1;\n"
        "timestep = 0.5;\n"
        "qm_prog = ase-dftbplus;\n"
        "slakos = custom;\n"
        "slakos_path = custom-slakos;\n"
        "start_file = start.rst;\n"
    )
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate custom-slakos.in --format=json
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "PQ accepted a missing Slater-Koster directory")
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "Slater-Koster directory.*does not exist")
        message(FATAL_ERROR "Unexpected Slater-Koster path diagnostic: ${output}")
    endif()
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate custom-slakos.in --scope=portable --format=json
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Portable validation required local Slater-Koster files: ${output}")
    endif()
    file(MAKE_DIRECTORY "${qm_without_descriptor_dir}/custom-slakos")
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate custom-slakos.in --format=json
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "PQ rejected an existing Slater-Koster directory: ${output}")
    endif()
endif()

file(
    WRITE "${qm_without_descriptor_dir}/cell-list.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "${qm_script_setting}\n"
    "cell-list = on;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate cell-list.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a cell list for a pure QM simulation")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "Cell lists are not available for pure QM")
    message(FATAL_ERROR "Unexpected pure-QM cell-list diagnostic: ${output}")
endif()

file(
    WRITE "${qm_without_descriptor_dir}/bad-hubbard.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = ase-dftbplus;\n"
    "hubbard_derivs = H:0.1junk;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate bad-hubbard.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a Hubbard derivative with trailing text")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
if(NOT diagnostic_line EQUAL 5)
    message(FATAL_ERROR "Hubbard derivative diagnostic lost its line: ${output}")
endif()

if(NOT EXPECTED_SHARED OR EXPECTED_SINGULARITY)
    file(
        WRITE "${qm_without_descriptor_dir}/name-only-script.in"
        "jobtype = qm-md;\n"
        "nstep = 1;\n"
        "timestep = 0.5;\n"
        "qm_prog = dftbplus;\n"
        "qm_script = dftbplus_periodic_stress;\n"
        "start_file = start.rst;\n"
    )
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate name-only-script.in --format=json
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "PQ accepted qm_script in this build mode")
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "requires.*qm_script_full_path")
        message(FATAL_ERROR "Unexpected build-specific script diagnostic: ${output}")
    endif()
endif()

file(
    WRITE "${qm_without_descriptor_dir}/npt.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "${qm_script_setting}\n"
    "start_file = start.rst;\n"
    "manostat = berendsen;\n"
    "pressure = 1.0;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate npt.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted pressure coupling without a moldescriptor")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "Moldescriptor file.*does not exist")
    message(FATAL_ERROR "Unexpected QM NPT descriptor diagnostic: ${output}")
endif()
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate npt.in --scope=portable --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation required a local moldescriptor: ${output}")
endif()
file(
    COPY
    "${VALIDATION_FIXTURE_DIR}/moldescriptor.dat"
    DESTINATION "${qm_without_descriptor_dir}"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate npt.in --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ required guff.dat for pure-QM NPT: ${output}")
endif()

set(missing_default_dir "${VALIDATION_WORK_DIR}/missing-default")
file(MAKE_DIRECTORY "${missing_default_dir}")
file(
    COPY
    "${VALIDATION_FIXTURE_DIR}/run.in"
    "${VALIDATION_FIXTURE_DIR}/start.rst"
    DESTINATION "${missing_default_dir}"
)
run_pq_in(
    "${missing_default_dir}"
    output error result
    --validate run.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a missing default input file")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "Moldescriptor file.*does not exist")
    message(FATAL_ERROR "Unexpected default file diagnostic: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/portable-missing-reference.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "start_file = missing-start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate portable-missing-reference.in --format=json --scope=portable
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation required a referenced file: ${output}")
endif()

file(
    WRITE "${qm_without_descriptor_dir}/ase-xtb.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = ase-xtb;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate ase-xtb.in --scope=portable --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation rejected an ASE input: ${output}")
endif()
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate ase-xtb.in --format=json
)
if(EXPECTED_ASE)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "ASE build rejected an ASE input: ${output}")
    endif()
else()
    if(result EQUAL 0)
        message(FATAL_ERROR "Non-ASE build accepted an installed ASE input")
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "requires ASE support")
        message(FATAL_ERROR "Unexpected ASE capability diagnostic: ${output}")
    endif()
endif()

file(
    WRITE "${qm_without_descriptor_dir}/fennol-model.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = fennol;\n"
    "fennol_model_path = missing.fnx;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate fennol-model.in --scope=portable --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation required a local FeNNol model: ${output}")
endif()
if(EXPECTED_ASE)
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate fennol-model.in --format=json
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "PQ accepted a missing FeNNol model file")
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "FeNNol model file.*does not exist")
        message(FATAL_ERROR "Unexpected FeNNol model diagnostic: ${output}")
    endif()
endif()

file(
    WRITE "${qm_without_descriptor_dir}/mace-model.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = mace;\n"
    "mace_model = custom;\n"
    "mace_model_path = missing.model;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate mace-model.in --scope=portable --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation required a local MACE model: ${output}")
endif()
if(EXPECTED_ASE)
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate mace-model.in --format=json
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "PQ accepted a missing local MACE model file")
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "MACE model file.*does not exist")
        message(FATAL_ERROR "Unexpected MACE model diagnostic: ${output}")
    endif()

    file(
        WRITE "${qm_without_descriptor_dir}/mace-model-url.in"
        "jobtype = qm-md;\n"
        "nstep = 1;\n"
        "timestep = 0.5;\n"
        "qm_prog = mace;\n"
        "mace_model = custom;\n"
        "mace_model_path = https://example.org/model.model;\n"
        "start_file = start.rst;\n"
    )
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate mace-model-url.in --format=json
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "PQ rejected a remote MACE model URL: ${output}")
    endif()
endif()

file(
    WRITE "${qm_without_descriptor_dir}/portable-threeob.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = ase-dftbplus;\n"
    "slakos = 3ob;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate portable-threeob.in --scope=portable --format=json
)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Portable validation rejected built-in Slater-Koster input: ${output}")
endif()
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate portable-threeob.in --format=json
)
if(EXPECTED_ASE)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "ASE build could not resolve built-in 3ob data: ${output}")
    endif()
else()
    if(result EQUAL 0)
        message(FATAL_ERROR "Non-ASE build accepted installed 3ob validation")
    endif()
endif()

file(
    WRITE "${qm_without_descriptor_dir}/contradictory-script.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "qm_script = dftbplus_periodic_stress;\n"
    "qm_script_full_path = runner;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate contradictory-script.in --scope=portable --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "Portable validation accepted contradictory QM scripts")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "mutually exclusive")
    message(FATAL_ERROR "Unexpected QM script diagnostic: ${output}")
endif()

file(
    WRITE "${qm_without_descriptor_dir}/missing-script.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = dftbplus;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate missing-script.in --scope=portable --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "Portable validation accepted a missing QM script")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "No qm_script provided")
    message(FATAL_ERROR "Unexpected missing QM script diagnostic: ${output}")
endif()

file(
    WRITE "${qm_without_descriptor_dir}/wrong-script.in"
    "jobtype = qm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "qm_prog = pyscf;\n"
    "qm_script = dftbplus_periodic_stress;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${qm_without_descriptor_dir}"
    output error result
    --validate wrong-script.in --scope=portable --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a bundled script for the wrong QM program")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "not available for pyscf")
    message(FATAL_ERROR "Unexpected QM script compatibility diagnostic: ${output}")
endif()

if(EXPECTED_SHARED AND NOT EXPECTED_SINGULARITY)
    file(
        WRITE "${qm_without_descriptor_dir}/turbomole-template.in"
        "jobtype = qm-md;\n"
        "nstep = 1;\n"
        "timestep = 0.5;\n"
        "qm_prog = turbomole;\n"
        "qm_script = turbomole_rimp2;\n"
        "start_file = start.rst;\n"
    )
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate turbomole-template.in --scope=portable --format=json
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Portable validation required a Turbomole template: ${output}")
    endif()
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate turbomole-template.in --format=json
    )
    if(result EQUAL 0)
        message(FATAL_ERROR "PQ accepted Turbomole without tm_define.template")
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "Required QM working file.*tm_define.template")
        message(FATAL_ERROR "Unexpected Turbomole template diagnostic: ${output}")
    endif()
    file(WRITE "${qm_without_descriptor_dir}/tm_define.template" "")
    run_pq_in(
        "${qm_without_descriptor_dir}"
        output error result
        --validate turbomole-template.in --format=json
    )
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "PQ rejected a complete Turbomole setup: ${output}")
    endif()
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/portable-missing-topology.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = on;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate portable-missing-topology.in --scope=portable --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "Portable validation accepted force-field input without topology")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "Topology file needed")
    message(FATAL_ERROR "Unexpected topology dependency diagnostic: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/portable-missing-mshake.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "force-field = off;\n"
    "shake = mshake;\n"
    "topology_file = missing.top;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate portable-missing-mshake.in --scope=portable --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "Portable validation accepted M-SHAKE without its file")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "M-SHAKE file needed")
    message(FATAL_ERROR "Unexpected M-SHAKE dependency diagnostic: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/bad-keyword.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "not_a_keyword = 1;\n"
    "timestep = 0.5;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate bad-keyword.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted an unknown keyword")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "Invalid JSON validation wrote to stderr: ${error}")
endif()
string(JSON validation_valid GET "${output}" valid)
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(validation_valid)
    message(FATAL_ERROR "Invalid input reported valid: ${output}")
endif()
if(NOT diagnostic_line EQUAL 3)
    message(FATAL_ERROR "Unexpected diagnostic line: ${diagnostic_line}")
endif()
if(NOT diagnostic_message MATCHES "Invalid keyword")
    message(FATAL_ERROR "Unexpected validation diagnostic: ${diagnostic_message}")
endif()

run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate bad-keyword.in
)
if(result EQUAL 0 OR NOT output STREQUAL "")
    message(FATAL_ERROR "Text validation did not reject the bad keyword")
endif()
string(REGEX MATCHALL "line 3" line_mentions "${error}")
list(LENGTH line_mentions line_mention_count)
if(NOT line_mention_count EQUAL 1)
    message(FATAL_ERROR "Text diagnostic repeated its line number: ${error}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/bad-number.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = nope;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate bad-number.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a malformed number")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_line EQUAL 3)
    message(FATAL_ERROR "Malformed number lost its line: ${output}")
endif()
if(NOT diagnostic_message MATCHES "Invalid value.*timestep")
    message(FATAL_ERROR "Unexpected numeric diagnostic: ${diagnostic_message}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/trailing-unit.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5fs;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate trailing-unit.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a numeric token with trailing text")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
if(NOT diagnostic_line EQUAL 3)
    message(FATAL_ERROR "Trailing-text diagnostic lost its line: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/fractional-integer.in"
    "jobtype = mm-md;\n"
    "nstep = 1.5;\n"
    "timestep = 0.5;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate fractional-integer.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a fractional integer")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
if(NOT diagnostic_line EQUAL 2)
    message(FATAL_ERROR "Fractional-integer diagnostic lost its line: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/unsafe-zero.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate unsafe-zero.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a zero timestep")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_line EQUAL 3)
    message(FATAL_ERROR "Unsafe value lost its line: ${output}")
endif()
if(NOT diagnostic_message MATCHES "Time step must be finite")
    message(FATAL_ERROR "Unexpected unsafe value diagnostic: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/zero-density.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "density = 0;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate zero-density.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a zero density")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
if(NOT diagnostic_line EQUAL 4)
    message(FATAL_ERROR "Zero-density diagnostic lost its line: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/zero-constraint-iterations.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "shake-iter = 0;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate zero-constraint-iterations.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted zero maximum constraint iterations")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
if(NOT diagnostic_line EQUAL 4)
    message(FATAL_ERROR "Constraint-iteration diagnostic lost its line: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/missing-timestep.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate missing-timestep.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted an MD input without timestep")
endif()
string(JSON diagnostic_line_type TYPE "${output}" diagnostics 0 line)
if(NOT diagnostic_line_type STREQUAL "NULL")
    message(FATAL_ERROR "Semantic diagnostic invented a line: ${output}")
endif()

foreach(decay_strategy IN ITEMS constant-decay exponential-decay)
    file(
        WRITE "${VALIDATION_WORK_DIR}/missing-${decay_strategy}.in"
        "jobtype = mm-opt;\n"
        "nstep = 1;\n"
        "learning-rate-strategy = ${decay_strategy};\n"
        "force-field = off;\n"
        "start_file = start.rst;\n"
    )
    run_pq_in(
        "${VALIDATION_WORK_DIR}"
        output error result
        --validate missing-${decay_strategy}.in --format=json
    )
    if(result EQUAL 0)
        message(
            FATAL_ERROR
            "PQ accepted ${decay_strategy} without learning-rate-decay"
        )
    endif()
    string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
    if(NOT diagnostic_message MATCHES "requires learning-rate-decay")
        message(
            FATAL_ERROR
            "Unexpected learning-rate decay diagnostic: ${output}"
        )
    endif()
endforeach()

file(
    WRITE "${VALIDATION_WORK_DIR}/unimplemented-line-search.in"
    "jobtype = mm-opt;\n"
    "nstep = 1;\n"
    "learning-rate-strategy = linesearch-wolfe;\n"
    "force-field = off;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate unimplemented-line-search.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted the unimplemented line search")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "not yet implemented")
    message(FATAL_ERROR "Unexpected line search diagnostic: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/overlapping-learning-rate-bounds.in"
    "jobtype = mm-opt;\n"
    "nstep = 1;\n"
    "learning-rate-strategy = constant;\n"
    "min-learning-rate = 0.5;\n"
    "max-learning-rate = 0.5;\n"
    "force-field = off;\n"
    "start_file = start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate overlapping-learning-rate-bounds.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted overlapping learning-rate bounds")
endif()
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_message MATCHES "minimum learning rate.*greater or equal")
    message(FATAL_ERROR "Unexpected learning-rate bounds diagnostic: ${output}")
endif()

file(
    WRITE "${VALIDATION_WORK_DIR}/missing-reference.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "start_file = missing\\\"start.rst;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate missing-reference.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a missing referenced file")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
string(JSON diagnostic_message GET "${output}" diagnostics 0 message)
if(NOT diagnostic_line EQUAL 4)
    message(FATAL_ERROR "Missing reference lost its line: ${output}")
endif()
if(NOT diagnostic_message MATCHES "missing.*start.rst")
    message(FATAL_ERROR "JSON escaping changed the diagnostic: ${output}")
endif()

file(MAKE_DIRECTORY "${VALIDATION_WORK_DIR}/start-directory")
file(
    WRITE "${VALIDATION_WORK_DIR}/directory-reference.in"
    "jobtype = mm-md;\n"
    "nstep = 1;\n"
    "timestep = 0.5;\n"
    "start_file = start-directory;\n"
)
run_pq_in(
    "${VALIDATION_WORK_DIR}"
    output error result
    --validate directory-reference.in --format=json
)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ accepted a directory as a referenced file")
endif()
string(JSON diagnostic_line GET "${output}" diagnostics 0 line)
if(NOT diagnostic_line EQUAL 4)
    message(FATAL_ERROR "Directory reference lost its line: ${output}")
endif()
