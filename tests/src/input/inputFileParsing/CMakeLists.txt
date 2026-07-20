set(source_files

    testInputFileParser.cpp

    testCelllistParser.cpp
    testConstraintsParser.cpp
    testConvergenceParser.cpp
    testCoulombLongRangeParser.cpp
    testFilesParser.cpp
    testGeneralParser.cpp
    testIntegratorParser.cpp
    testManostatParser.cpp
    testMMParser.cpp
    testOptParser.cpp
    testOutputParser.cpp
    testQMParser.cpp
    testResetKineticsParser.cpp
    testRingPolymerParser.cpp
    testSimulationBoxParser.cpp
    testThermostatParser.cpp
    testTimingsParser.cpp
    testVirialParser.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/input
        ${PROJECT_SOURCE_DIR}/tests/include/macros
    )
    target_link_libraries(${test_name}
        PRIVATE
        PQ_input
        gtest
        gmock
        pq_test_main
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )

    set_property(TEST ${test_name} PROPERTY LABELS inputFileParser)
endforeach()

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_inputFileParser
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;inputFileParser"
        OUTPUT_PATH "coverage"
    )
endif()