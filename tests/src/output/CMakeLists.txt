set(source_files
    testStdoutOutput.cpp
    testLogOutput.cpp
    testRstFileOutput.cpp
    testTrajectoryOutput.cpp
    testInfoOutput.cpp
    testEnergyOutput.cpp
    testOutput.cpp
    testMomentumOutput.cpp
    testRingPolymerRestartFileOutput.cpp
    testRingPolymerTrajectoryOutput.cpp
    testOptOutput.cpp
    testTimingsOutput.cpp
)

# Per-test extra include dirs and libs (defaults to empty for the legacy tests).
set(testOptOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/settings
)
set(testOptOutput_extra_libs
    optimization
    convergence
    settings
)

set(testTimingsOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/timings
)
set(testTimingsOutput_extra_libs
    timings
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
        ${PROJECT_SOURCE_DIR}/tests/include/output
        ${${test_name}_extra_includes}
    )
    target_link_libraries(${test_name}
        PRIVATE
        output
        simulationBox
        ${${test_name}_extra_libs}
        gtest
        gmock
        pq_test_main
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )

    set_property(TEST ${test_name} PROPERTY LABELS output)
endforeach()

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_output
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        # DEPENDENCIES ${CMAKE_BUILD_DIR}/src/output
        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;output"
        OUTPUT_PATH "coverage"
    )
endif()
