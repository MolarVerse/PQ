option(BUILD_WITH_MPI "Build with MPI" OFF)

if(BUILD_WITH_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DWITH_MPI)
    include_directories(${MPI_INCLUDE_PATH})
endif()