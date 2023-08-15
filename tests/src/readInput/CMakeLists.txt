set(source_files
    testRstFileReader.cpp
    testMoldescriptorReader.cpp
    testGuffDatReader.cpp
    testInputFileReader.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests/include/readInput ${PROJECT_SOURCE_DIR}/tests/include/macros)
    target_link_libraries(${test_name} PRIVATE readInput potential gtest gtest_main gmock gmock_main)
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
endforeach()

add_subdirectory(inputFileParsing)
add_subdirectory(topologyReader)
add_subdirectory(parameterFileReader)
add_subdirectory(rstFileSection)