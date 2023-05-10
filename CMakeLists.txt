cmake_minimum_required(VERSION 3.7...3.26)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(PIMD-QMCF VERSION 1.0
    DESCRIPTION "Path Integral Molecular Dynamics - Quantum Charge Field"
    LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/.cmake)

if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("Release build.")
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -Werror -pedantic -O3")
else()
    message("Debug build.")
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -g -pg -fprofile-arcs -ftest-coverage -O0 -coverage")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

set(CODE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(apps)

include(doxygen)
include(testing)