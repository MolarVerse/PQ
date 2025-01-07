# *********************
# minimum cmake version
# *********************
cmake_minimum_required(VERSION 3.7...3.26)

# # ****************
# # set cmake policy
# # ****************
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# ***********************
# project initializeation
# ***********************
project(PQ
    DESCRIPTION "MolarVerse - PQ"
    LANGUAGES CXX C)

# **************************
# add .cmake to include path
# **************************
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.cmake)

# ********
# includes
# ********
include(config) # include default settins
include(getGitVersion) # include get git version

# **************
# check compiler
# **************
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "Insufficient gcc version (need >= 13.0)")
    endif()
else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with GNU.")
endif()

if(WIN32)
    set(CMAKE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/result")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIRECTORY})
else()
    # *****************
    # set install rpath
    # *****************
    SET(CMAKE_INSTALL_RPATH
        "${CMAKE_INSTALL_PREFIX}/lib/;${CMAKE_INSTALL_PREFIX}/lib/tools/;"
    )
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# ***************
# cmake cxx flags
# ***************
set(CMAKE_CXX_STANDARD 20)

# set default flags
set(CMAKE_CXX_FLAGS "-std=c++20 -Wall -Wextra")

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-use-linker-plugin")
endif()

# add flags dependent on build type
if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("\n   Release build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

    if(NOT BUILD_WITH_KOKKOS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]el[Ww]ith[Dd]ebug")
    message("\n   Release with Debug build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -coverage -pg -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wshadow -Wdouble-promotion -Wformat=2 -Wundef")
elseif(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
    message("\n   Debug build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -coverage -pg -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wshadow -Wdouble-promotion -Wformat=2 -Wundef")
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_COMPILE_VERSION_=\\\"${GIT_VERSION}\\\"")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}\n")

# ************************
# include dir for c++ code
# ************************
set(CODE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

include(eigen)

if(BUILD_WITH_MPI)
    include(mpi)
endif()

if(BUILD_WITH_KOKKOS)
    include(kokkos)
endif()

if(BUILD_WITH_PYTHON_BINDINGS)
    include(python_bindings)
endif()

if(BUILD_WITH_ASE)
    include(ase)
    include(slakos)
endif()

if(BUILD_WITH_TESTS)
    include(testing)
endif()

add_subdirectory(src)

if(BUILD_WITH_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(apps)

if(BUILD_WITH_DOC)
    include(doxygen)
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

if(BUILD_WITH_IWYU)
    include(iwyu)
endif()

if(BUILD_WITH_BENCHMARKING)
    include(benchmark)
endif()
