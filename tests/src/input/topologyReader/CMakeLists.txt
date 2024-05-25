set(source_files
    testTopologyReader.cpp
    testTopologySection.cpp
    testShakeSection.cpp
    testBondSection.cpp
    testAngleSection.cpp
    testDihedralSection.cpp
    testImproperDihedralSection.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/input
    )
    target_link_libraries(${test_name}
        PRIVATE
        topologyFileReader
        gtest
        gmock
        pq_test_main
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_name}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )

    set_property(TEST ${test_name} PROPERTY LABELS topologyReader)
endforeach()

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_topologyReader
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;topologyReader"
        OUTPUT_PATH "coverage"
    )
endif()