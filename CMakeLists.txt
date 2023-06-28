# minimum cmake version
cmake_minimum_required(VERSION 3.7...3.26)

# set cmake policy
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(PIMD-QMCF VERSION 1.0
    DESCRIPTION "Path Integral Molecular Dynamics - Quantum Charge Field"
    LANGUAGES CXX C)

# set if shared or static libraries
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
endif()

SET(CMAKE_INSTALL_RPATH
    "${CMAKE_INSTALL_PREFIX}/lib/;${CMAKE_INSTALL_PREFIX}/lib/tools/;"
)

# SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/tools/lib)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/.cmake)

if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("Release build.")

    # set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Wextra -Werror -pedantic -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Wextra -Ofast -ffast-math -flto -march=native -mtune=native -g")
else()
    message("Debug build.")

    # set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Wextra -g -ftest-coverage -O0 -coverage")
    set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Wextra -g -pg -fprofile-arcs -ftest-coverage -O0 -coverage") # for gprof
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

set(CODE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(mpi)
include(doxygen)
include(testing)

add_subdirectory(tests)
include(benchmarking)
add_subdirectory(benchmarks)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tools)