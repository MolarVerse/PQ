set(source_files
    testBoxSection.cpp
    testNoseHooverSection.cpp
    testStepCountSection.cpp
    testAtomSection.cpp
    testRstFileReader.cpp

    testParserBox.cpp
    testParserIntegrator.cpp
    testParserTimings.cpp
    testParserOutput.cpp
    testParserGeneral.cpp
    testInputFileParser.cpp
    testInputFileReader.cpp

    testMoldescriptorReader.cpp

    testPostProcessSetup.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/tests/include/setup ${PROJECT_SOURCE_DIR}/tests/include/utilities)
    target_link_libraries(${test_name} PRIVATE setup gtest gtest_main gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endforeach()