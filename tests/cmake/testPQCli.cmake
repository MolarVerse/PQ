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
string(JSON kokkos GET "${output}" build kokkos)
string(JSON python_bindings GET "${output}" build python_bindings)
string(JSON python_embedding GET "${output}" build python_embedding)
string(JSON t_relaxation GET "${output}" input parameters t_relaxation default)
string(JSON random_seed_max GET "${output}" input parameters random_seed maximum)
if(NOT schema STREQUAL "pq.capabilities")
    message(FATAL_ERROR "Unexpected capabilities schema: ${schema}")
endif()
if(NOT schema_version EQUAL 1)
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
if(NOT kokkos STREQUAL "${EXPECTED_KOKKOS}")
    message(FATAL_ERROR "Unexpected Kokkos capability: ${kokkos}")
endif()
if(NOT python_bindings STREQUAL "${EXPECTED_PYTHON_BINDINGS}")
    message(FATAL_ERROR "Unexpected Python bindings capability: ${python_bindings}")
endif()
if(NOT python_embedding STREQUAL "${EXPECTED_PYTHON_EMBEDDING}")
    message(FATAL_ERROR "Unexpected Python embedding capability: ${python_embedding}")
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
