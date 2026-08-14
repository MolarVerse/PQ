function(run_pq output_var error_var result_var)
    execute_process(
        COMMAND "${PQ_EXECUTABLE}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    set(${output_var} "${output}" PARENT_SCOPE)
    set(${error_var} "${error}" PARENT_SCOPE)
    set(${result_var} "${result}" PARENT_SCOPE)
endfunction()

function(assert_input_array field expected_items)
    string(JSON actual_count LENGTH "${output}" input "${field}")
    list(LENGTH expected_items expected_count)
    if(NOT actual_count EQUAL expected_count)
        message(
            FATAL_ERROR
            "Unexpected ${field} count: ${actual_count}; expected ${expected_count}"
        )
    endif()

    math(EXPR last_index "${expected_count} - 1")
    foreach(index RANGE 0 ${last_index})
        string(JSON actual_item GET "${output}" input "${field}" ${index})
        list(GET expected_items ${index} expected_item)
        if(NOT actual_item STREQUAL expected_item)
            message(
                FATAL_ERROR
                "Unexpected ${field}[${index}]: ${actual_item}; expected ${expected_item}"
            )
        endif()
    endforeach()
endfunction()

run_pq(output error result --help)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ --help returned ${result}: ${error}")
endif()
if(NOT output MATCHES "^Usage: PQ <input_file>")
    message(FATAL_ERROR "PQ --help did not print usage: ${output}")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "PQ --help wrote to stderr: ${error}")
endif()

run_pq(output error result --version)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ --version returned ${result}: ${error}")
endif()
if(NOT output STREQUAL "PQ ${EXPECTED_VERSION}\n")
    message(
        FATAL_ERROR
        "PQ --version printed '${output}', expected 'PQ ${EXPECTED_VERSION}'"
    )
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "PQ --version wrote to stderr: ${error}")
endif()

run_pq(output error result --capabilities=json)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "PQ --capabilities=json returned ${result}: ${error}")
endif()
if(NOT error STREQUAL "")
    message(FATAL_ERROR "PQ --capabilities=json wrote to stderr: ${error}")
endif()
string(JSON schema GET "${output}" schema)
string(JSON schema_version GET "${output}" schema_version)
string(JSON version GET "${output}" version)
string(JSON ase GET "${output}" build ase)
string(JSON mpi GET "${output}" build mpi)
string(JSON python_bindings GET "${output}" build python_bindings)
string(JSON python_embedding GET "${output}" build python_embedding)
string(JSON shared GET "${output}" build shared)
string(JSON static_build GET "${output}" build static)
string(JSON singularity GET "${output}" build singularity)
string(
    JSON validation_schema
    GET "${output}" cli input_validation schema
)
string(
    JSON validation_schema_version
    GET "${output}" cli input_validation schema_version
)
string(
    JSON validation_formats
    GET "${output}" cli input_validation formats
)
string(
    JSON validation_scopes
    GET "${output}" cli input_validation scopes
)
string(JSON script_mode GET "${output}" input external_qm script_mode)
string(
    JSON dftbplus_script
    GET "${output}" input external_qm programs dftbplus scripts 0 name
)
string(
    JSON dftbplus_required_keyword
    GET "${output}" input external_qm programs dftbplus scripts 0
        required_file_keywords 0
)
string(
    JSON pyscf_script_count
    LENGTH "${output}" input external_qm programs pyscf scripts
)
string(
    JSON pyscf_hf_script
    GET "${output}" input external_qm programs pyscf scripts 0 name
)
string(
    JSON pyscf_mp2_script
    GET "${output}" input external_qm programs pyscf scripts 1 name
)
string(
    JSON turbomole_script
    GET "${output}" input external_qm programs turbomole scripts 0 name
)
string(
    JSON turbomole_required_file
    GET "${output}" input external_qm programs turbomole scripts 0
        required_working_files 0
)
string(JSON nstep_min GET "${output}" input parameters nstep minimum)
string(JSON nstep_max GET "${output}" input parameters nstep maximum)
string(
    JSON timestep_min
    GET "${output}" input parameters timestep exclusive_minimum
)
string(
    JSON output_freq_min
    GET "${output}" input parameters output_freq minimum
)
string(
    JSON output_freq_max
    GET "${output}" input parameters output_freq maximum
)
string(JSON temp_min GET "${output}" input parameters temp minimum)
string(
    JSON temp_ramp_frequency_min
    GET "${output}" input parameters temp_ramp_frequency minimum
)
string(JSON t_relaxation GET "${output}" input parameters t_relaxation default)
string(
    JSON t_relaxation_min
    GET "${output}" input parameters t_relaxation exclusive_minimum
)
string(
    JSON t_relaxation_max
    GET "${output}" input parameters t_relaxation maximum
)
string(
    JSON t_relaxation_source
    GET "${output}" input parameters t_relaxation minimum_from parameter
)
string(
    JSON t_relaxation_factor
    GET "${output}" input parameters t_relaxation minimum_from factor
)
string(
    JSON friction_min
    GET "${output}" input parameters friction minimum
)
string(
    JSON friction_max
    GET "${output}" input parameters friction maximum
)
string(
    JSON chain_length_min
    GET "${output}" input parameters nh-chain_length minimum
)
string(
    JSON coupling_frequency_max
    GET "${output}" input parameters coupling_frequency maximum
)
string(
    JSON p_relaxation_source
    GET "${output}" input parameters p_relaxation minimum_from parameter
)
string(
    JSON p_relaxation_factor
    GET "${output}" input parameters p_relaxation minimum_from factor
)
string(
    JSON compressibility_min
    GET "${output}" input parameters compressibility minimum
)
string(
    JSON density_min
    GET "${output}" input parameters density exclusive_minimum
)
string(JSON rcoulomb_min GET "${output}" input parameters rcoulomb minimum)
string(JSON random_seed_max GET "${output}" input parameters random_seed maximum)
if(NOT schema STREQUAL "pq.capabilities")
    message(FATAL_ERROR "Unexpected capabilities schema: ${schema}")
endif()
if(NOT schema_version EQUAL 2)
    message(FATAL_ERROR "Unexpected capabilities schema version: ${schema_version}")
endif()
if(NOT version STREQUAL "${EXPECTED_VERSION}")
    message(FATAL_ERROR "Unexpected capabilities PQ version: ${version}")
endif()
if(NOT ase STREQUAL "${EXPECTED_ASE}")
    message(FATAL_ERROR "Unexpected ASE capability: ${ase}")
endif()
if(NOT mpi STREQUAL "${EXPECTED_MPI}")
    message(FATAL_ERROR "Unexpected MPI capability: ${mpi}")
endif()
if(NOT python_bindings STREQUAL "${EXPECTED_PYTHON_BINDINGS}")
    message(FATAL_ERROR "Unexpected Python bindings capability: ${python_bindings}")
endif()
if(NOT python_embedding STREQUAL "${EXPECTED_PYTHON_EMBEDDING}")
    message(FATAL_ERROR "Unexpected Python embedding capability: ${python_embedding}")
endif()
if(NOT shared STREQUAL "${EXPECTED_SHARED}")
    message(FATAL_ERROR "Unexpected shared-build capability: ${shared}")
endif()
if(static_build STREQUAL shared)
    message(FATAL_ERROR "Unexpected static-build capability: ${static_build}")
endif()
if(NOT singularity STREQUAL "${EXPECTED_SINGULARITY}")
    message(FATAL_ERROR "Unexpected Singularity capability: ${singularity}")
endif()
if(NOT validation_schema STREQUAL "pq.validation")
    message(FATAL_ERROR "Unexpected validation schema: ${validation_schema}")
endif()
if(NOT validation_schema_version EQUAL 1)
    message(
        FATAL_ERROR
        "Unexpected validation schema version: ${validation_schema_version}"
    )
endif()
if(NOT validation_formats MATCHES "json")
    message(FATAL_ERROR "JSON validation format is not advertised")
endif()
if(
    NOT validation_scopes MATCHES "portable"
    OR NOT validation_scopes MATCHES "installed"
)
    message(FATAL_ERROR "Validation scopes are incomplete")
endif()
if(static_build OR EXPECTED_SINGULARITY)
    set(expected_script_mode full_path_only)
else()
    set(expected_script_mode bundled_or_full_path)
endif()
if(NOT script_mode STREQUAL expected_script_mode)
    message(FATAL_ERROR "Unexpected external-QM script mode: ${script_mode}")
endif()
if(NOT dftbplus_script STREQUAL "dftbplus_periodic_stress")
    message(FATAL_ERROR "Unexpected DFTB+ script: ${dftbplus_script}")
endif()
if(NOT dftbplus_required_keyword STREQUAL "dftb_file")
    message(
        FATAL_ERROR
        "Unexpected DFTB+ file requirement: ${dftbplus_required_keyword}"
    )
endif()
if(
    NOT pyscf_script_count EQUAL 2
    OR NOT pyscf_hf_script STREQUAL "pyscf_hf.py"
    OR NOT pyscf_mp2_script STREQUAL "pyscf_mp2.py"
)
    message(FATAL_ERROR "Unexpected PySCF script catalog")
endif()
if(NOT turbomole_script STREQUAL "turbomole_ricc2")
    message(FATAL_ERROR "Unexpected Turbomole script: ${turbomole_script}")
endif()
if(NOT turbomole_required_file STREQUAL "tm_define.template")
    message(
        FATAL_ERROR
        "Unexpected Turbomole file requirement: ${turbomole_required_file}"
    )
endif()

assert_input_array(
    job_types
    "mm-md;mm-hessian;mm-opt;qm-md;qm-rpmd"
)
set(expected_qm_programs "dftbplus;pyscf;turbomole")
if(EXPECTED_ASE)
    list(
        APPEND expected_qm_programs
        ase_dftbplus ase_xtb fennol mace mace_mp mace_off
    )
endif()
assert_input_array(qm_programs "${expected_qm_programs}")
assert_input_array(
    thermostats
    "none;berendsen;velocity_rescaling;langevin;nh-chain"
)
assert_input_array(manostats "none;berendsen;stochastic_rescaling")
assert_input_array(
    pressure_isotropies
    "isotropic;xy;xz;yz;anisotropic;full_anisotropic"
)

if(NOT t_relaxation EQUAL 0.1)
    message(FATAL_ERROR "Unexpected t_relaxation default: ${t_relaxation}")
endif()
if(NOT nstep_min EQUAL 1 OR NOT nstep_max EQUAL 2147483647)
    message(FATAL_ERROR "Unexpected nstep bounds")
endif()
if(
    NOT output_freq_min EQUAL 0
    OR NOT output_freq_max EQUAL 2147483647
)
    message(FATAL_ERROR "Unexpected output frequency bounds")
endif()
if(NOT timestep_min EQUAL 0 OR NOT temp_min EQUAL 0)
    message(FATAL_ERROR "Unexpected timestep or temperature bound")
endif()
if(NOT temp_ramp_frequency_min EQUAL 1)
    message(FATAL_ERROR "Unexpected temperature ramp frequency minimum")
endif()
if(
    NOT t_relaxation_min EQUAL 0
    OR NOT t_relaxation_source STREQUAL "timestep"
    OR NOT t_relaxation_factor EQUAL 0.001
)
    message(FATAL_ERROR "Unexpected thermostat relaxation constraint")
endif()
if(
    NOT p_relaxation_source STREQUAL "timestep"
    OR NOT p_relaxation_factor EQUAL 0.001
)
    message(FATAL_ERROR "Unexpected manostat relaxation constraint")
endif()
if(
    NOT friction_min EQUAL 0
    OR NOT chain_length_min EQUAL 1
    OR NOT compressibility_min EQUAL 0
    OR NOT density_min EQUAL 0
    OR NOT rcoulomb_min EQUAL 0
)
    message(FATAL_ERROR "Unexpected scalar parameter bound")
endif()
foreach(
    upper_bound
    IN ITEMS
        t_relaxation_max
        friction_max
        coupling_frequency_max
)
    if(NOT ${upper_bound} GREATER 0)
        message(FATAL_ERROR "Missing positive maximum for ${upper_bound}")
    endif()
endforeach()
if(NOT random_seed_max EQUAL 4294967295)
    message(FATAL_ERROR "Unexpected random seed maximum: ${random_seed_max}")
endif()

run_pq(output error result --unknown)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ --unknown returned success")
endif()
if(NOT output STREQUAL "")
    message(FATAL_ERROR "PQ --unknown wrote to stdout: ${output}")
endif()
if(NOT error MATCHES "Unknown option: --unknown")
    message(FATAL_ERROR "PQ --unknown did not report the option: ${error}")
endif()

run_pq(output error result)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ without an input returned success")
endif()
if(NOT output STREQUAL "")
    message(FATAL_ERROR "PQ without an input wrote to stdout: ${output}")
endif()
if(NOT error MATCHES "No input file specified")
    message(FATAL_ERROR "PQ without an input did not report the error: ${error}")
endif()

run_pq(output error result missing-input.in)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ with a missing input returned success")
endif()
if(NOT output STREQUAL "")
    message(FATAL_ERROR "PQ with a missing input wrote to stdout: ${output}")
endif()
if(NOT error MATCHES "File not found")
    message(FATAL_ERROR "PQ with a missing input did not report the error: ${error}")
endif()

run_pq(output error result missing-input.in extra)
if(result EQUAL 0)
    message(FATAL_ERROR "PQ with an extra argument returned success")
endif()
if(NOT output STREQUAL "")
    message(FATAL_ERROR "PQ with an extra argument wrote to stdout: ${output}")
endif()
if(NOT error MATCHES "Unexpected argument: extra")
    message(FATAL_ERROR "PQ did not report the extra argument: ${error}")
endif()
