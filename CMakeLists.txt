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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    if(BUILD_WITH_NATIVE)
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
        elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
        endif()
    endif()

    if(BUILD_WITH_LTO AND NOT BUILD_WITH_KOKKOS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]el[Ww]ith[Dd]ebug")
    message("\n   Release with Debug build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wshadow -Wdouble-promotion -Wformat=2 -Wundef")
    if(UNIX AND NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
    message("\n   Debug build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wshadow -Wdouble-promotion -Wformat=2 -Wundef")
    if(UNIX AND NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    endif()
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_COMPILE_VERSION_=\\\"${GIT_VERSION}\\\"")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}\n")

# *********************************
# optional build-speed accelerators
# *********************************
# use ccache as compiler launcher if available (greatly speeds up rebuilds)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "ccache found - using it as compiler launcher: ${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# use a faster linker (mold, then lld) if available (speeds up linking)
if(UNIX AND NOT APPLE)
    find_program(MOLD_PROGRAM mold)
    find_program(LLD_PROGRAM ld.lld)
    if(MOLD_PROGRAM)
        message(STATUS "mold found - using it as linker")
        add_link_options("-fuse-ld=mold")
    elseif(LLD_PROGRAM)
        message(STATUS "lld found - using it as linker")
        add_link_options("-fuse-ld=lld")
    endif()
endif()

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
