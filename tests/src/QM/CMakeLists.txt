add_executable(testExternalQMRunner
    testExternalQMRunner.cpp
)

target_include_directories(testExternalQMRunner
    PRIVATE
    ${PROJECT_SOURCE_DIR}/tests/include/macros
)

target_link_libraries(testExternalQMRunner
    PRIVATE
    externalQM
    gtest
    gmock
    pq_test_main
)

target_precompile_headers(testExternalQMRunner REUSE_FROM pq_test_pch)
target_link_libraries(testExternalQMRunner PRIVATE pq_test_pch)

add_test(
    NAME testExternalQMRunner
    COMMAND testExternalQMRunner
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
)

set_property(TEST testExternalQMRunner PROPERTY LABELS QM)
