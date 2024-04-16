add_library(PQ_mpi
    mpi.cpp
)

target_include_directories(PQ_mpi
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/mpi
)

if(BUILD_WITH_MPI)
    target_link_libraries(PQ_mpi
        PRIVATE
        MPI::MPI_C
    )
endif()

install(TARGETS PQ_mpi
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)