add_library(pimd_qmcf_mpi
    mpi.cpp
)

target_include_directories(pimd_qmcf_mpi
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/mpi
)

if(BUILD_WITH_MPI)
    target_link_libraries(pimd_qmcf_mpi
        PRIVATE
        MPI::MPI_C
    )
endif()

install(TARGETS pimd_qmcf_mpi
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)