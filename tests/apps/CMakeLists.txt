set(source_files
    testCommandLineArgs.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/apps/commandLine
        ${PROJECT_SOURCE_DIR}/tests/include/input
        ${PROJECT_SOURCE_DIR}/tests/include/macros
    )
    target_link_libraries(${test_name}
        PRIVATE
        gtest
        gmock
        pq_test_main
        exceptions
        commandLineArgs
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )
    target_precompile_headers(${test_name} REUSE_FROM pq_test_pch)
    target_link_libraries(${test_name} PRIVATE pq_test_pch)

    set_property(TEST ${test_name} PROPERTY LABELS apps)
endforeach()

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_apps
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;input"
        OUTPUT_PATH "coverage"
    )
endif()
