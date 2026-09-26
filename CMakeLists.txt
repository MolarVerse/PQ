# *********************
# minimum cmake version
# *********************
cmake_minimum_required(VERSION 3.20...3.26)

# ************************************
# for generating compile_commands.json
# ************************************
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
# Avoid colliding with Google Benchmark's GetGitVersion.cmake on
# case-insensitive filesystems.
include(pqGitVersion) # include get git version

# **************
# check compiler
# **************
set(PQ_COMPILER_IS_GNU FALSE)
set(PQ_COMPILER_IS_CLANG FALSE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(PQ_COMPILER_IS_GNU TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(PQ_COMPILER_IS_CLANG TRUE)
endif()

if(PQ_COMPILER_IS_GNU)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "Insufficient gcc version (need >= 13.0)")
    endif()
elseif(PQ_COMPILER_IS_CLANG)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.0)
        message(FATAL_ERROR "Insufficient clang version (need >= 16.0)")
    endif()
else()
    message(WARNING "You are using an unsupported compiler! Compilation has only been tested with GNU and Clang.")
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
    if(APPLE)
        set(CMAKE_INSTALL_RPATH "@loader_path/../lib;@loader_path")
    else()
        set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN")
    endif()
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()

# ***************
# cmake cxx flags
# ***************
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set default flags
# Warning flags are collected in PQ_WARNING_FLAGS and only applied later via
# add_compile_options(), after all third-party code (googletest, benchmark,
# pybind11, ...) has been added. This keeps them out of external targets.
set(PQ_WARNING_FLAGS -Wall -Wextra)

if(PQ_COMPILER_IS_GNU)
    list(APPEND PQ_WARNING_FLAGS -Wswitch-enum -Werror=switch-enum)
endif()

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-use-linker-plugin")
endif()

# add flags dependent on build type
if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease")
    message("\n   Release build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    if(BUILD_WITH_NATIVE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=native")
        endif()
    endif()

    if(BUILD_WITH_LTO)
        include(CheckIPOSupported)
        check_ipo_supported(
            RESULT ltoSupported
            OUTPUT ltoError
            LANGUAGES CXX
        )
        if(NOT ltoSupported)
            message(FATAL_ERROR "LTO requested but unavailable: ${ltoError}")
        endif()
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "^[Rr]el[Ww]ith[Dd]ebug")
    message("\n   Release with Debug build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -coverage -fprofile-arcs -ftest-coverage")
    list(APPEND PQ_WARNING_FLAGS -Wshadow -Wdouble-promotion -Wformat=2 -Wundef)
    if(PQ_COMPILER_IS_GNU)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    endif()
elseif(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug")
    message("\n   Debug build   \n")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og -coverage -fprofile-arcs -ftest-coverage")
    list(APPEND PQ_WARNING_FLAGS
        -Wpedantic -Werror
        -Wundef
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wcast-align
        -Wformat=2
        -Wnull-dereference
        -Wdouble-promotion
        -Wimplicit-fallthrough
        -Woverloaded-virtual
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wmisleading-indentation
        # lifetime / safety
        -Wuninitialized
        -Winit-self
    )

    # list(APPEND PQ_WARNING_FLAGS -Wno-error=deprecated-declarations)
    # list(APPEND PQ_WARNING_FLAGS -Wno-unknown-warning-option)
    # list(APPEND PQ_WARNING_FLAGS -Wno-c2y-extensions)
    if(PQ_COMPILER_IS_GNU)
        add_compile_options(-pg)
        # GNU specific warnings
        list(APPEND PQ_WARNING_FLAGS
            -Wuseless-cast
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wmaybe-uninitialized
        )
    endif()
    if(PQ_COMPILER_IS_CLANG)
        message(STATUS ">>> PQ: adding -ftime-trace")
        add_compile_options(-ftime-trace)
        #TODO: these should be removed as soon as linearalgebra is migrated to mstd
        list(APPEND PQ_WARNING_FLAGS -Wno-gnu-anonymous-struct -Wno-nested-anon-types)
    endif()
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_COMPILE_VERSION_=\\\"${GIT_VERSION}\\\"")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "PQ_WARNING_FLAGS: ${PQ_WARNING_FLAGS}\n")

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

find_package(Git QUIET)

if(Git_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)

    if(GIT_SUBMODULE AND NOT EXISTS "${PROJECT_SOURCE_DIR}/external/mstd/include/")
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE GIT_SUBMOD_RESULT)

        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

include(eigen)

if(BUILD_WITH_MPI)
    include(mpi)
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

if(BUILD_WITH_BENCHMARKING)
    include(benchmarking)
endif()

# ******************************************************
# project warning policy - applied to PQ targets only
# all third-party code has to be added above this line
# ******************************************************
foreach(flag IN LISTS PQ_WARNING_FLAGS)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${flag}>")
endforeach()

# external headers are marked SYSTEM to suppress their warnings
set_target_properties(eigen PROPERTIES
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    "$<TARGET_PROPERTY:eigen,INTERFACE_INCLUDE_DIRECTORIES>"
)

add_library(mstd INTERFACE)
target_include_directories(mstd SYSTEM INTERFACE ${PROJECT_SOURCE_DIR}/external/mstd/include)

add_subdirectory(src)

if(BUILD_WITH_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(apps)

if(BUILD_WITH_PERF_BENCH)
    add_subdirectory(benchmarks/perf)
endif()

if(BUILD_WITH_DOCS)
    add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

if(BUILD_WITH_IWYU)
    include(iwyu)
endif()

if(BUILD_WITH_BENCHMARKING)
    enable_testing()
    add_subdirectory(benchmarks)
endif()
