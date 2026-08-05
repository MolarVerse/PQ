if(NOT DEFINED STAGING_PREFIX OR STAGING_PREFIX STREQUAL "" OR
   STAGING_PREFIX STREQUAL "/")
    message(FATAL_ERROR "A safe staging prefix is required")
endif()

file(REMOVE_RECURSE "${STAGING_PREFIX}")

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --prefix "${STAGING_PREFIX}"
        --component references
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

if(NOT DEFINED REFERENCE_TEST_EXECUTABLE OR
   NOT EXISTS "${REFERENCE_TEST_EXECUTABLE}")
    message(FATAL_ERROR "Reference test executable is required")
endif()

foreach(reference IN ITEMS
        pq.ref
        pq.ref.bib
        dftbplus.ref
        velocity_verlet.ref)
    if(NOT EXISTS "${STAGING_PREFIX}/share/PQ/references/${reference}")
        message(FATAL_ERROR "Reference data ${reference} was not installed")
    endif()
endforeach()

get_filename_component(
    reference_test_binary_dir
    "${REFERENCE_TEST_EXECUTABLE}"
    DIRECTORY
)
get_filename_component(
    reference_test_asset_root
    "${reference_test_binary_dir}"
    DIRECTORY
)
set(reference_test_share "${reference_test_asset_root}/share/PQ/references")

file(REMOVE_RECURSE "${reference_test_share}")
file(MAKE_DIRECTORY "${reference_test_share}")
file(
    COPY "${STAGING_PREFIX}/share/PQ/references/"
    DESTINATION "${reference_test_share}"
)

set(reference_marker "STAGED_PQ_REFERENCE_DATA")
file(
    APPEND
    "${reference_test_share}/pq.ref"
    "\n${reference_marker}\n"
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}" -E env
        "PQ_TEST_EXPECTED_REFERENCE_MARKER=${reference_marker}"
        "${REFERENCE_TEST_EXECUTABLE}"
        "--gtest_filter=TestReferencesOutput.writeReferencesFileEmitsHeaderAndBibtexBanner"
    WORKING_DIRECTORY "${STAGING_PREFIX}"
    TIMEOUT 30
    RESULT_VARIABLE reference_test_result
    OUTPUT_VARIABLE reference_test_output
    ERROR_VARIABLE reference_test_error
)
if(NOT reference_test_result EQUAL 0)
    message(
        FATAL_ERROR
        "Executable-relative reference test failed: "
        "${reference_test_output} ${reference_test_error}"
    )
endif()

file(REMOVE_RECURSE "${reference_test_share}")
file(REMOVE_RECURSE "${STAGING_PREFIX}")
