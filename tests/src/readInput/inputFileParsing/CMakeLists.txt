set(source_files

    # testInputFileParser.cpp
    # testParserSimulationBox.cpp
    # testParserCelllist.cpp
    # testParserConstraints.cpp
    # testParserGeneral.cpp
    # testParserIntegrator.cpp
    # testParserManostat.cpp
    # testParserOutput.cpp
    # testParserResetKinetics.cpp
    testParserThermostat.cpp
    testParserTimings.cpp
    testParserTopology.cpp
    testParserVirial.cpp

    # testParserCoulombLongRange.cpp
    # testParserNonCoulomb.cpp
    # testParserParameterFile.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/include/readInput ${PROJECT_SOURCE_DIR}/tests/include/macros)
    target_link_libraries(${test_name} PRIVATE readInput config gtest gtest_main gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endforeach()