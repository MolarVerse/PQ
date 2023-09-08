set(source_files
    testConstraints.cpp
    testBondConstraint.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/constraints
        ${PROJECT_SOURCE_DIR}/tests/include/macros
    )
    target_link_libraries(${test_name}
        PRIVATE
        constraints
        settings
        utilities
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

    set_property(TEST ${test_name} PROPERTY LABELS constraints)
endforeach()

if(${CMAKE_BUILD_TYPE} MATCHES "^[Dd]ebug")
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_constraints
        EXCLUDE ${EXCLUDE_FOR_GCOVR}
        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;constraints"
        OUTPUT_PATH "coverage"
    )
endif()