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

# **********************************************************
# BULID_WITH_MACE - MACE is a library for machine learning
# that is used in the ML module of the
# *********************************************************
option(BUILD_WITH_MACE "Build with MACE" ON)

# **********************
# BUILD WITH SINGULARITY
# **********************
option(BUILD_WITH_SINGULARITY "Build with Singularity" OFF)