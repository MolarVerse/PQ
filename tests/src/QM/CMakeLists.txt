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

add_test(
    NAME testExternalQMRunner
    COMMAND testExternalQMRunner
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
)

set_property(TEST testExternalQMRunner PROPERTY LABELS QM)
