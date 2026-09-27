set(PQ_TEST_PCH_HEADERS
    <gtest/gtest.h>
    <format>
    <memory>
    <string>
    <vector>
)

if(BUILD_WITH_ASE)
    list(APPEND PQ_TEST_PCH_HEADERS
        <pybind11/embed.h>
        <pybind11/numpy.h>
        <pybind11/pybind11.h>
        <pybind11/stl.h>
    )
endif()

add_library(pq_test_pch OBJECT dummy.cpp)
target_precompile_headers(pq_test_pch PRIVATE ${PQ_TEST_PCH_HEADERS})

target_link_libraries(pq_test_pch
        PUBLIC
        gtest
        gmock
)

if(BUILD_WITH_ASE)
    target_link_libraries(pq_test_pch
        PUBLIC
        pybind11::embed
        pybind11::module
    )
endif()

# Create cleanup script at configure time
file(WRITE "${CMAKE_BINARY_DIR}/cleanup_gcda.cmake"
    "file(GLOB_RECURSE GCDA_FILES \"${CMAKE_BINARY_DIR}/**/*.gcda\")\n"
    "if(GCDA_FILES)\n"
    "    file(REMOVE \${GCDA_FILES})\n"
    "endif()\n"
)

# Add cleanup as a test that runs first
add_test(NAME cleanup-gcda 
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_BINARY_DIR}/cleanup_gcda.cmake"
)
set_tests_properties(cleanup-gcda PROPERTIES
    FIXTURES_SETUP "gcda-cleanup"
)

# Helper function
function(add_test_with_gcda_cleanup test_name test_executable)
    target_include_directories(${test_name}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/tests/include/macros
    )
    add_test(
        NAME ${test_name}
        COMMAND ${test_executable}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    )
    set_tests_properties(${test_name} PROPERTIES
        FIXTURES_REQUIRED "gcda-cleanup"
    )
endfunction()

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(regression)
