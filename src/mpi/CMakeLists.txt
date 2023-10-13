add_library(pimd_qmcf_mpi
    mpi.cpp
)

target_include_directories(pimd_qmcf_mpi
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/mpi
)

install(TARGETS pimd_qmcf_mpi
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/
)