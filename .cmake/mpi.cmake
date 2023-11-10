find_package(MPI REQUIRED)
add_definitions(-DWITH_MPI)
add_definitions(-DOMPI_SKIP_MPICXX)
include_directories(${MPI_INCLUDE_PATH})