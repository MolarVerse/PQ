set(source_files
    testResetKinetics.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
        ${PROJECT_SOURCE_DIR}/tests/include/resetKinetics
    )
    target_link_libraries(${test_name}
        PRIVATE
        resetKinetics
        physicalData
        molsys
        gtest
        pq_test_main
        gmock
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )
    target_precompile_headers(${test_name} REUSE_FROM pq_test_pch)
    target_link_libraries(${test_name} PRIVATE pq_test_pch)

    set_property(TEST ${test_name} PROPERTY LABELS resetKinetics)
endforeach()

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_resetKinetics
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        # DEPENDENCIES ${CMAKE_BUILD_DIR}/src/resetKinetics
        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;resetKinetics"
        OUTPUT_PATH "coverage"
    )
endif()
