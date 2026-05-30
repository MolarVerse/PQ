# **********
# BUILD TYPE
# **********
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebug" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_BUILD_TYPES "Debug" "RelWithDebug" "Release")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_BUILD_TYPES})

# *****************
# shared vs. static
# *****************
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# **************
# INSTALL PREFIX
# **************
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/install)
endif()

# *****************
# BUILD WITH KOKKOS
# *****************
option(BUILD_WITH_KOKKOS "Build with Kokkos" OFF)

# *****************
# BUILD WITH TESTS
# *****************
option(BUILD_WITH_TESTS "Build tests" ON)

# **************
# BUILD WITH MPI
# **************
option(BUILD_WITH_MPI "Build with MPI" OFF)

# ***********
# BUILD TOOLS
# ***********
option(BUILD_WITH_PYTHON_BINDINGS "Build with python bindings" OFF)

# ***************
# BUILD WITH IWYU
# ***************
option(BUILD_WITH_IWYU "Build with include-what-you-use" OFF)

# ****************
# BUILD WITH GCOVR
# ****************
option(BUILD_WITH_GCOVR "Build with gcovr" OFF)

# ***************
# BUILD WITH DOCS
# ***************
option(BUILD_WITH_DOCS "Build documentation" ON)

# ***********************
# BUILD WITH BENCHMARKING
# ***********************
option(BUILD_WITH_BENCHMARKING "Build benchmarking" OFF)

# **************
# BULID_WITH_ASE
# **************
option(BUILD_WITH_ASE "Build with ASE" ON)

# **********************
# BUILD WITH SINGULARITY
# **********************
option(BUILD_WITH_SINGULARITY "Build with Singularity" OFF)

# **************
# BUILD WITH LTO
# **************
option(BUILD_WITH_LTO "Build Release with link-time optimization (-flto)" OFF)

# *****************
# BUILD WITH NATIVE
# *****************
# Tune Release for the build machine's CPU (-march=native). Turn OFF for
# portable/cacheable builds (e.g. CI), where -march=native would otherwise
# produce binaries that crash on a different CPU.
option(BUILD_WITH_NATIVE "Optimize Release for the build machine's CPU (-march=native)" ON)

# *********************
# BUILD WITH PERF BENCH
# *********************
# Build the fixed-work performance-regression benchmark (benchmarks/perf),
# used by CI to gate on instruction-count regressions via callgrind.
option(BUILD_WITH_PERF_BENCH "Build the performance-regression benchmark" OFF)