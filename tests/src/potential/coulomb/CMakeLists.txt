set(source_files
    testCoulombPotential.cpp
    testCoulombShiftedPotential.cpp
    testCoulombWolf.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/potential
    )
    target_link_libraries(${test_name}
        PRIVATE
        readInput
        potential
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

    set_property(TEST ${test_name} PROPERTY LABELS coulombPotential)
endforeach()

include(CodeCoverage)
setup_target_for_coverage_gcovr_html(
    NAME coverage_coulombPotential
    EXCLUDE ${EXCLUDE_FOR_GCOVR}

    EXECUTABLE "ctest"
    EXECUTABLE_ARGS "-L;coulombPotential"
    OUTPUT_PATH "coverage"
)