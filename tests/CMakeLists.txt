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

add_subdirectory(src)
add_subdirectory(regression)
