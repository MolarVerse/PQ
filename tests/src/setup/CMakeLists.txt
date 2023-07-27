set(source_files
    testBoxSection.cpp
    testNoseHooverSection.cpp
    testStepCountSection.cpp
    testAtomSection.cpp
    testRstFileReader.cpp
    testMoldescriptorReader.cpp
    testGuffDatReader.cpp
    testPostProcessSetup.cpp
    testInputFileReader.cpp
    testTopologyReader.cpp
    testShakeSection.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/include/setup)
    target_link_libraries(${test_name} PRIVATE setup gtest gtest_main gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endforeach()

add_subdirectory(InputFileParsing)