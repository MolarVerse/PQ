cmake_minimum_required(VERSION 3.7...3.26)
include(.cmake/documentation.cmake)

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
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -Wextra -g")
endif()

set(CODE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)
add_subdirectory(apps)

message(${CMAKE_MODULE_PATH})

include(documentation)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    enable_testing()
    find_package(GTest CONFIG)  # use googletest
    if(NOT ${GTest_FOUND})
        message("GTest not found. Fetching...")
        include(FetchContent)
        FetchContent_Declare(
            gtest
            GIT_REPOSITORY "https://github.com/google/googletest"
            GIT_TAG        main
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(gtest)
    endif()

    list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
    add_subdirectory(tests)
endif()