set(source_files
    testIntraNonBondedMap.cpp
    testIntraNonBonded.cpp
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
    )
    target_link_libraries(${test_name} PRIVATE
        intraNonBonded
        simulationBox
        potential
        linearAlgebra
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

    set_property(TEST ${test_name} PROPERTY LABELS intraNonBonded)
endforeach()

if(NOT ${CMAKE_BUILD_TYPE} MATCHES "^[Rr]elease")
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_intraNonBonded
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        # DEPENDENCIES ${CMAKE_BUILD_DIR}/src/intraNonBonded
        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;intraNonBonded"
        OUTPUT_PATH "coverage"
    )
endif()