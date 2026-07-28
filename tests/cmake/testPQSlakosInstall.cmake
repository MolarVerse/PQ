if(NOT DEFINED STAGING_PREFIX OR STAGING_PREFIX STREQUAL "" OR
   STAGING_PREFIX STREQUAL "/")
    message(FATAL_ERROR "A safe staging prefix is required")
endif()

file(REMOVE_RECURSE "${STAGING_PREFIX}")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --prefix "${STAGING_PREFIX}"
        --component slakos
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

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --prefix "${STAGING_PREFIX}"
        --component references
    RESULT_VARIABLE reference_install_result
    OUTPUT_VARIABLE reference_install_output
    ERROR_VARIABLE reference_install_error
)
if(NOT reference_install_result EQUAL 0)
    message(
        FATAL_ERROR
        "Could not stage reference data: "
        "${reference_install_output} ${reference_install_error}"
    )
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

foreach(reference IN ITEMS 3ob.ref 3ob.ref.bib matsci.ref matsci.ref.bib)
    if(NOT EXISTS "${STAGING_PREFIX}/share/PQ/references/${reference}")
        message(FATAL_ERROR "Reference data ${reference} was not installed")
    endif()
endforeach()

file(READ "${STAGING_PREFIX}/share/PQ/references/3ob.ref" threeob_references)
if(NOT threeob_references MATCHES "10.1021/ct300849w")
    message(FATAL_ERROR "3ob reference data is incomplete")
endif()

file(READ "${STAGING_PREFIX}/share/PQ/references/matsci.ref" matsci_references)
if(NOT matsci_references MATCHES "10.1002/zaac.200500051")
    message(FATAL_ERROR "matsci reference data is incomplete")
endif()

get_filename_component(test_binary_dir "${SLAKOS_TEST_EXECUTABLE}" DIRECTORY)
get_filename_component(test_asset_root "${test_binary_dir}/.." ABSOLUTE)
set(test_slakos_root "${test_asset_root}/share/PQ/slakos")
file(MAKE_DIRECTORY
    "${test_slakos_root}/3ob/skfiles"
    "${test_slakos_root}/matsci/skfiles"
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "PQ_TEST_EXPECTED_SLAKOS_ROOT=${test_slakos_root}"
        "${SLAKOS_TEST_EXECUTABLE}"
        "--gtest_filter=QMSettingsTest.ResolvesBundledSlakos"
    RESULT_VARIABLE lookup_result
    OUTPUT_VARIABLE lookup_output
    ERROR_VARIABLE lookup_error
)

file(REMOVE_RECURSE "${test_slakos_root}")
file(REMOVE_RECURSE "${STAGING_PREFIX}")

if(NOT lookup_result EQUAL 0)
    message(
        FATAL_ERROR
        "Installed Slakos lookup failed: ${lookup_output} ${lookup_error}"
    )
endif()
