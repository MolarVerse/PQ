if(NOT DEFINED STAGING_PREFIX OR STAGING_PREFIX STREQUAL "" OR
   STAGING_PREFIX STREQUAL "/")
    message(FATAL_ERROR "A safe staging prefix is required")
endif()

function(run_staged_validation input_file)
    execute_process(
        COMMAND "${STAGED_EXECUTABLE}"
                --validate "${input_file}" --format=json
        WORKING_DIRECTORY "${STAGING_PREFIX}/work"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error
    )
    if(NOT result EQUAL 0)
        message(
            FATAL_ERROR
            "Staged PQ rejected ${input_file}: ${output} ${error}"
        )
    endif()
    string(JSON valid GET "${output}" valid)
    if(NOT valid)
        message(FATAL_ERROR "Staged validation reported invalid: ${output}")
    endif()
endfunction()

file(REMOVE_RECURSE "${STAGING_PREFIX}")
file(MAKE_DIRECTORY "${STAGING_PREFIX}/work")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --prefix "${STAGING_PREFIX}"
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
    message(
        FATAL_ERROR
        "Could not stage PQ: ${install_output} ${install_error}"
    )
endif()

get_filename_component(pq_executable_name "${PQ_EXECUTABLE}" NAME)
set(STAGED_EXECUTABLE "${STAGING_PREFIX}/bin/${pq_executable_name}")
if(NOT EXISTS "${STAGED_EXECUTABLE}")
    message(FATAL_ERROR "PQ executable was not installed")
endif()

foreach(script IN ITEMS
        dftbplus_periodic_stress
        pyscf_hf.py
        pyscf_mp2.py
        turbomole_rimp2)
    if(NOT EXISTS "${STAGING_PREFIX}/share/PQ/scripts/${script}")
        message(FATAL_ERROR "QM script ${script} was not installed")
    endif()
endforeach()

foreach(reference IN ITEMS
        pq.ref
        pq.ref.bib
        3ob.ref
        3ob.ref.bib
        matsci.ref
        matsci.ref.bib)
    if(NOT EXISTS "${STAGING_PREFIX}/share/PQ/references/${reference}")
        message(FATAL_ERROR "Reference data ${reference} was not installed")
    endif()
endforeach()

file(READ "${STAGING_PREFIX}/share/PQ/references/3ob.ref" threeob_references)
foreach(doi IN ITEMS
        "10.1021/ct300849w"
        "10.1021/ct401002w"
        "10.1021/jp506557r"
        "10.1021/ct5009137")
    if(NOT threeob_references MATCHES "${doi}")
        message(FATAL_ERROR "3ob reference data is missing ${doi}")
    endif()
endforeach()

file(READ "${STAGING_PREFIX}/share/PQ/references/matsci.ref" matsci_references)
foreach(marker IN ITEMS
        "TU Dresden"
        "10.1002/zaac.200500051"
        "10.1021/nn700184k"
        "10.1016/j.susc.2008.01.035"
        "10.1021/jp8110343"
        "10.3139/146.110337"
        "Jardillier")
    if(NOT matsci_references MATCHES "${marker}")
        message(FATAL_ERROR "matsci reference data is missing ${marker}")
    endif()
endforeach()

file(GLOB installed_libraries "${STAGING_PREFIX}/lib/*")
if(NOT installed_libraries)
    message(FATAL_ERROR "PQ libraries were not installed")
endif()

foreach(slakos_set IN ITEMS 3ob matsci)
    set(slakos_root "${STAGING_PREFIX}/share/PQ/slakos/${slakos_set}")
    foreach(metadata IN ITEMS LICENSE README RELEASE.md CHANGELOG.md)
        if(NOT EXISTS "${slakos_root}/${metadata}")
            message(
                FATAL_ERROR
                "${slakos_set} metadata ${metadata} was not installed"
            )
        endif()
    endforeach()
    foreach(pair IN ITEMS H-H C-C O-O)
        if(NOT EXISTS "${slakos_root}/skfiles/${pair}.skf")
            message(
                FATAL_ERROR
                "${slakos_set} parameter ${pair}.skf was not installed"
            )
        endif()
    endforeach()

    file(
        GLOB source_parameters
        "${SLAKOS_SOURCE_DIR}/${slakos_set}/skfiles/*.skf"
    )
    file(GLOB installed_parameters "${slakos_root}/skfiles/*.skf")
    list(LENGTH source_parameters source_parameter_count)
    list(LENGTH installed_parameters installed_parameter_count)
    if(NOT installed_parameter_count EQUAL source_parameter_count)
        message(
            FATAL_ERROR
            "${slakos_set} installed ${installed_parameter_count} of "
            "${source_parameter_count} parameter files"
        )
    endif()
endforeach()

execute_process(
    COMMAND "${STAGED_EXECUTABLE}" --capabilities=json
    RESULT_VARIABLE capabilities_result
    OUTPUT_VARIABLE capabilities_output
    ERROR_VARIABLE capabilities_error
)
if(NOT capabilities_result EQUAL 0)
    message(
        FATAL_ERROR
        "Installed PQ capabilities failed: "
        "${capabilities_output} ${capabilities_error}"
    )
endif()

file(
    COPY
    "${MM_FIXTURE_DIR}/run.in"
    "${MM_FIXTURE_DIR}/start.rst"
    "${MM_FIXTURE_DIR}/moldescriptor.dat"
    "${MM_FIXTURE_DIR}/guff.dat"
    DESTINATION "${STAGING_PREFIX}/work"
)

foreach(slakos_set IN ITEMS 3ob matsci)
    file(
        WRITE "${STAGING_PREFIX}/work/${slakos_set}.in"
        "jobtype = qm-md;\n"
        "nstep = 1;\n"
        "timestep = 0.5;\n"
        "qm_prog = ase-dftbplus;\n"
        "slakos = ${slakos_set};\n"
        "start_file = start.rst;\n"
    )
    run_staged_validation("${slakos_set}.in")
endforeach()

execute_process(
    COMMAND "${STAGED_EXECUTABLE}" run.in
    WORKING_DIRECTORY "${STAGING_PREFIX}/work"
    TIMEOUT 30
    RESULT_VARIABLE simulation_result
    OUTPUT_VARIABLE simulation_output
    ERROR_VARIABLE simulation_error
)
if(NOT simulation_result EQUAL 0)
    message(
        FATAL_ERROR
        "Installed PQ simulation failed: "
        "${simulation_output} ${simulation_error}"
    )
endif()
if(NOT simulation_output MATCHES "PQ ended normally")
    message(FATAL_ERROR "Installed PQ did not report normal completion")
endif()

set(reference_output "${STAGING_PREFIX}/work/smoke.ref")
if(NOT EXISTS "${reference_output}")
    message(FATAL_ERROR "Installed PQ did not write smoke.ref")
endif()
file(READ "${reference_output}" reference_contents)
foreach(expected IN ITEMS "PQ Software" "BIBTEX ENTRIES")
    if(NOT reference_contents MATCHES "${expected}")
        message(FATAL_ERROR "smoke.ref is missing ${expected}")
    endif()
endforeach()

file(REMOVE_RECURSE "${STAGING_PREFIX}")
