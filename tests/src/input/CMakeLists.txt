set(source_files
    testRestartFileReader.cpp
    testMoldescriptorReader.cpp
    testGuffDatReader.cpp
    testInputFileReader.cpp
    testIntraNonBondedReader.cpp
    testCommandLineArgs.cpp
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
        input
        restartFileReader
        potential
        engine
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

    set_property(TEST ${test_name} PROPERTY LABELS input)
endforeach()

if(${CMAKE_BUILD_TYPE} MATCHES "^[Dd]ebug")
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_input
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;input"
        OUTPUT_PATH "coverage"
    )
endif()

add_subdirectory(inputFileParsing)
add_subdirectory(topologyReader)
add_subdirectory(parameterFileReader)
add_subdirectory(restartFileSection)