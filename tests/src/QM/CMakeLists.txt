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

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME testOrcaScript
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrcaScript.py
            ${PROJECT_SOURCE_DIR}/src/QM/scripts/orca_rimp2.py
            ${PROJECT_SOURCE_DIR}/examples/orca/orca.template)
set_property(TEST testOrcaScript PROPERTY LABELS QM)

add_test(NAME testOrcaScfScripts
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testOrcaScfScripts.py
            ${PROJECT_SOURCE_DIR}/src/QM/scripts
            ${PROJECT_SOURCE_DIR}/examples/orca)
set_property(TEST testOrcaScfScripts PROPERTY LABELS QM)

find_program(PERL_EXECUTABLE perl)
if(PERL_EXECUTABLE)
    add_test(NAME testTurbomoleScripts
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/testTurbomoleScripts.py
                ${PROJECT_SOURCE_DIR}/src/QM/scripts)
    set_property(TEST testTurbomoleScripts PROPERTY LABELS QM)
endif()
