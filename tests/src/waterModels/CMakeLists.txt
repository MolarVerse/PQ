add_executable(testWaterModels testWaterModels.cpp)

target_include_directories(testWaterModels
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/waterModels/intra
    ${PROJECT_SOURCE_DIR}/include/waterModels/inter
    ${PROJECT_SOURCE_DIR}/tests/include/macros
)

target_link_libraries(testWaterModels
    PRIVATE
    interWater
    potential
    simulationBox
    virial
    gtest
    gmock
    pq_test_main
)

add_test(
    NAME testWaterModels
    COMMAND testWaterModels
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
)

target_precompile_headers(testWaterModels REUSE_FROM pq_test_pch)
target_link_libraries(testWaterModels PRIVATE pq_test_pch)

set_property(TEST testWaterModels PROPERTY LABELS waterModels)

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_waterModels
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;waterModels"
        OUTPUT_PATH "coverage"
    )
endif()
