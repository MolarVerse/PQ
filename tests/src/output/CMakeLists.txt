set(source_files
    testStdoutOutput.cpp
    testLogOutput.cpp
    testRstFileOutput.cpp
    testTrajectoryOutput.cpp
    testInfoOutput.cpp
    testEnergyOutput.cpp
    testEnergyColumns.cpp
    testOutput.cpp
    testMomentumOutput.cpp
    testRingPolymerRestartFileOutput.cpp
    testRingPolymerTrajectoryOutput.cpp
    testOptOutput.cpp
    testTimingsOutput.cpp
    testBoxOutput.cpp
    testStressOutput.cpp
    testVirialOutput.cpp
    testRingPolymerEnergyOutput.cpp
    testReferencesOutput.cpp
    testFrame.cpp
    testTextBuffer.cpp
    testExtxyzOutput.cpp
    testH5mdOutput.cpp
    testAseTrajOutput.cpp
    testAtomicNumber.cpp
    testUlmWriter.cpp
)

# Per-test extra include dirs and libs (defaults to empty for the legacy tests).
set(testOptOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/opt/optimizer
    ${PROJECT_SOURCE_DIR}/include/opt/convergence
    ${PROJECT_SOURCE_DIR}/include/settings
)
set(testOptOutput_extra_libs
    optimization
    convergence
    settings
)

set(testTimingsOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/timings
)
set(testTimingsOutput_extra_libs
    timings
)

set(testBoxOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/box
)
set(testBoxOutput_extra_libs
    box
)

set(testStressOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/physicalData
)
set(testStressOutput_extra_libs
    physicalData
)

set(testVirialOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/physicalData
)
set(testVirialOutput_extra_libs
    physicalData
)

set(testRingPolymerEnergyOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/physicalData
)
set(testRingPolymerEnergyOutput_extra_libs
    physicalData
)

set(testReferencesOutput_extra_includes
    ${PROJECT_SOURCE_DIR}/include/output/references
    ${PROJECT_SOURCE_DIR}/include/settings
)
set(testReferencesOutput_extra_libs
    references
    settings
)

set(testFrame_extra_includes
    ${PROJECT_SOURCE_DIR}/include/box
    ${PROJECT_SOURCE_DIR}/include/thermostat
)
set(testFrame_extra_libs
    box
    thermostat
)

foreach(source_file ${source_files})
    get_filename_component(test_name ${source_file} NAME_WE)
    add_executable(${test_name} ${source_file})
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
        ${PROJECT_SOURCE_DIR}/tests/include/output
        ${${test_name}_extra_includes}
    )
    target_link_libraries(${test_name}
        PRIVATE
        output
        molsys
        ${${test_name}_extra_libs}
        gtest
        gmock
        pq_test_main
    )
    add_test_with_gcda_cleanup(${test_name} ${test_name})
    target_precompile_headers(${test_name} REUSE_FROM pq_test_pch)
    target_link_libraries(${test_name} PRIVATE pq_test_pch)

    set_property(TEST ${test_name} PROPERTY LABELS output)
endforeach()

# reads the H5MD file back with the HDF5 library, so it is built without the
# shared precompiled header
if(BUILD_WITH_HDF5)
    add_executable(testH5mdOutputContent testH5mdOutputContent.cpp)
    set_source_files_properties(testH5mdOutputContent.cpp PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
    )
    target_include_directories(testH5mdOutputContent
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include/box
    )
    target_link_libraries(testH5mdOutputContent
        PRIVATE
        output
        molsys
        box
        HDF5::HDF5
        gtest
        gmock
        pq_test_main
    )
    add_test_with_gcda_cleanup(testH5mdOutputContent testH5mdOutputContent)
    set_property(TEST testH5mdOutputContent PROPERTY LABELS output)

    add_executable(testHdf5Wrapper testHdf5Wrapper.cpp)
    set_source_files_properties(testHdf5Wrapper.cpp PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON
    )
    target_link_libraries(testHdf5Wrapper
        PRIVATE
        output
        HDF5::HDF5
        gtest
        gmock
        pq_test_main
    )
    add_test_with_gcda_cleanup(testHdf5Wrapper testHdf5Wrapper)
    set_property(TEST testHdf5Wrapper PROPERTY LABELS output)
endif()

if(${BUILD_WITH_GCOVR})
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_html(
        NAME coverage_output
        EXCLUDE ${EXCLUDE_FOR_GCOVR}

        # DEPENDENCIES ${CMAKE_BUILD_DIR}/src/output
        EXECUTABLE "ctest"
        EXECUTABLE_ARGS "-L;output"
        OUTPUT_PATH "coverage"
    )
endif()
