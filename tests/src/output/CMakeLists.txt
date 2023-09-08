set(source_files
    testStdoutOutput.cpp
    testLogOutput.cpp
    testRstFileOutput.cpp
    testTrajectoryOutput.cpp
    testInfoOutput.cpp
    testEnergyOutput.cpp
    testOutput.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
        ${PROJECT_SOURCE_DIR}/tests/include/output
    )
    target_link_libraries(${test_name}
        PRIVATE
        output
        simulationBox
        gtest
        gtest_main
        gmock
        gmock_main
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )

    set_property(TEST ${test_name} PROPERTY LABELS output)
endforeach()

if(${CMAKE_BUILD_TYPE} MATCHES "^[Dd]ebug)
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