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

add_test_with_gcda_cleanup(testExternalQMRunner testExternalQMRunner)

set_property(TEST testExternalQMRunner PROPERTY LABELS QM)

if(BUILD_WITH_ASE)
    add_executable(testAseQMRunner
        testAseQMRunner.cpp
    )

    target_link_libraries(testAseQMRunner
        PRIVATE
        aseQM
        gtest
        gmock
        pq_test_main
    )

    add_test_with_gcda_cleanup(testAseQMRunner testAseQMRunner)

    set_property(TEST testAseQMRunner PROPERTY LABELS QM)
endif()
