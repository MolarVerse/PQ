if(NOT DEFINED STAGING_PREFIX OR STAGING_PREFIX STREQUAL "" OR
   STAGING_PREFIX STREQUAL "/")
    message(FATAL_ERROR "A safe staging prefix is required")
endif()

file(REMOVE_RECURSE "${STAGING_PREFIX}")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --prefix "${STAGING_PREFIX}"
        --component qm-scripts
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

foreach(script IN ITEMS
        dftbplus_periodic_stress
        pyscf_hf.py
        pyscf_mp2.py
        turbomole_hf-dft
        turbomole_ricc2)
    if(NOT EXISTS "${STAGING_PREFIX}/share/PQ/scripts/${script}")
        message(FATAL_ERROR "QM script ${script} was not installed")
    endif()
endforeach()

get_filename_component(test_binary_dir "${SCRIPT_TEST_EXECUTABLE}" DIRECTORY)
get_filename_component(test_asset_root "${test_binary_dir}/.." ABSOLUTE)
set(test_script_dir "${test_asset_root}/share/PQ/scripts")
file(MAKE_DIRECTORY "${test_script_dir}")
file(
    COPY "${STAGING_PREFIX}/share/PQ/scripts/pyscf_hf.py"
    DESTINATION "${test_script_dir}"
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "PQ_TEST_EXPECTED_SCRIPT_DIR=${test_script_dir}"
        "${SCRIPT_TEST_EXECUTABLE}"
        "--gtest_filter=TestQMSetup.resolvesBundledQMScript"
    RESULT_VARIABLE lookup_result
    OUTPUT_VARIABLE lookup_output
    ERROR_VARIABLE lookup_error
)

file(REMOVE "${test_script_dir}/pyscf_hf.py")
file(REMOVE_RECURSE "${STAGING_PREFIX}")

if(NOT lookup_result EQUAL 0)
    message(
        FATAL_ERROR
        "Installed QM script lookup failed: ${lookup_output} ${lookup_error}"
    )
endif()
